import numpy as np
import einops
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as thf
import pytorch_lightning as pl
import torchvision
from latent_diffmodel.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from torchvision.utils import make_grid
from latent_diffmodel.modules.attention import SpatialTransformer
from latent_diffmodel.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, \
    AttentionBlock
from latent_diffmodel.models.diffusion.ddpm import LatentDiffusion
from latent_diffmodel.util import log_txt_as_img, exists, instantiate_from_config, default
from latent_diffmodel.models.diffusion.ddim import DDIMSampler
from latent_diffmodel.modules.ema import LitEma
from latent_diffmodel.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from latent_diffmodel.modules.diffusionmodules.model import Encoder
import lpips
from kornia import color


def disabled_train(self, mode=True):

    return self


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class SecretEncoder(nn.Module):
    def __init__(self, secret_len, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res * 3),
            nn.SiLU(),
            View(-1, 3, base_res, base_res),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base), 2 ** (log_resolution - log_base))),
            # chx16x16 -> chx256x256
            zero_module(conv_nd(2, 3, 3, 3, padding=1))
        )  # secret len -> ch x res x res

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None





class SecretEncoders(nn.Module):
    """join img emb with secret emb"""

    def __init__(self, secret_len, ch=3, base_res=16, resolution=64, emode='c3') -> None:
        super().__init__()
        assert emode in ['c3', 'c2', 'm3']

        if emode == 'c3':  # c3: concat c and x each has ch channels
            secret_ch = ch
            join_ch = 2 * ch
        elif emode == 'c2':  # c2: concat c (2) and x ave (1)
            secret_ch = 2
            join_ch = ch
        elif emode == 'm3':  # m3: multiply c (ch) and x (ch)
            secret_ch = ch
            join_ch = ch

            # m3: multiply c (ch) and x ave (1)
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.emode = emode
        self.resolution = resolution
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res * secret_ch),
            nn.SiLU(),
            View(-1, secret_ch, base_res, base_res),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base), 2 ** (log_resolution - log_base))),
            # chx16x16 -> chx256x256
        )  # secret len -> ch x res x res
        self.join_encoder = nn.Sequential(
            conv_nd(2, join_ch, join_ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, join_ch, ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, ch, ch, 3, padding=1),
            nn.SiLU()
        )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        if self.emode == 'c3':
            x = torch.cat([x, c], dim=1)
        elif self.emode == 'c2':
            x = torch.cat([x.mean(dim=1, keepdim=True), c], dim=1)
        elif self.emode == 'm3':
            x = x * c
        dx = self.join_encoder(x)
        dx = self.out_layer(dx)
        return dx, None






class ControlAE(pl.LightningModule):
    def __init__(self,
                 first_stage_key,
                 first_stage_config,
                 control_key,
                 control_config,
                 decoder_config,
                 loss_config,
                 noise_config='__none__',
                 use_ema=False,
                 secret_warmup=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.control_key = control_key
        self.first_stage_key = first_stage_key
        self.ae = instantiate_from_config(first_stage_config)
        self.control = instantiate_from_config(control_config)
        self.decoder = instantiate_from_config(decoder_config)
        if noise_config != '__none__':
            print('Using noise')
            self.noise = instantiate_from_config(noise_config)
        # copy weights from first stage
        self.control.copy_encoder_weight(self.ae)
        # freeze first stage
        self.ae.eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False

        self.loss_layer = instantiate_from_config(loss_config)

        # early training phase
        # self.fixed_input = True
        self.fixed_x = None
        self.fixed_img = None
        self.fixed_input_recon = None
        self.fixed_control = None
        self.register_buffer("fixed_input", torch.tensor(True))

        # secret warmup
        self.secret_warmup = secret_warmup
        self.secret_baselen = 2
        self.secret_len = control_config.params.secret_len
        if self.secret_warmup:
            assert self.secret_len == 2 ** (int(np.log2(self.secret_len)))

        self.use_ema = use_ema
        if self.use_ema:
            print('Using EMA')
            self.control_ema = LitEma(self.control)
            self.decoder_ema = LitEma(self.decoder)
            print(f"Keeping EMAs of {len(list(self.control_ema.buffers()) + list(self.decoder_ema.buffers()))}.")

        if ckpt_path != '__none__':
            self.init_from_ckpt(ckpt_path, ignore_keys=[])

    def get_warmup_secret(self, old_secret):

        if self.secret_warmup:
            bsz = old_secret.shape[0]
            nrepeats = self.secret_len // self.secret_baselen
            new_secret = torch.zeros((bsz, self.secret_baselen), dtype=torch.float).random_(0, 2).repeat_interleave(
                nrepeats, dim=1)
            return new_secret.to(old_secret.device)
        else:
            return old_secret

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.control_ema.store(self.control.parameters())
            self.decoder_ema.store(self.decoder.parameters())
            self.control_ema.copy_to(self.control)
            self.decoder_ema.copy_to(self.decoder)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.control_ema.restore(self.control.parameters())
                self.decoder_ema.restore(self.decoder.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.control_ema(self.control)
            self.decoder_ema(self.decoder)

    def compute_loss(self, pred, target):
        # return thf.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
        lpips_loss = self.lpips_loss(pred, target).mean(dim=[1, 2, 3])
        pred_yuv = color.rgb_to_yuv((pred + 1) / 2)
        target_yuv = color.rgb_to_yuv((target + 1) / 2)
        yuv_loss = torch.mean((pred_yuv - target_yuv) ** 2, dim=[2, 3])
        yuv_loss = 1.5 * torch.mm(yuv_loss, self.yuv_scales).squeeze(1)
        return lpips_loss + yuv_loss

    def forward(self, x, image, c):
        if self.control.__class__.__name__ == 'SecretEncoders':
            eps, posterior = self.control(x, c)
        else:
            eps, posterior = self.control(image, c)
        return x + eps, posterior

    @torch.no_grad()
    def get_input(self, batch, return_first_stage=False, bs=None):
        image = batch[self.first_stage_key]
        control = batch[self.control_key]
        control = self.get_warmup_secret(control)
        if bs is not None:
            image = image[:bs]
            control = control[:bs]
        else:
            bs = image.shape[0]
        # encode image 1st stage
        image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
        x = self.encode_first_stage(image).detach()
        image_rec = self.decode_first_stage(x).detach()

        # check if using fixed input (early training phase)
        # if self.training and self.fixed_input:
        if self.fixed_input:
            if self.fixed_x is None:  # first iteration
                print('[TRAINING]-using clear input image for now!')
                self.fixed_x = x.detach().clone()[:bs]
                self.fixed_img = image.detach().clone()[:bs]
                self.fixed_input_recon = image_rec.detach().clone()[:bs]
                self.fixed_control = control.detach().clone()[:bs]  # use for log_images with fixed_input option only
            x, image, image_rec = self.fixed_x, self.fixed_img, self.fixed_input_recon

        out = [x, control]
        if return_first_stage:
            out.extend([image, image_rec])

        return out

    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        image_rec = self.ae.decode(z)
        return image_rec

    def encode_first_stage(self, image):
        encoder_posterior = self.ae.encode(image)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def shared_step(self, batch):
        x, c, img, _ = self.get_input(batch, return_first_stage=True)
        # import pdb; pdb.set_trace()
        x, posterior = self(x, img, c)
        image_rec = self.decode_first_stage(x)
        # resize
        if img.shape[-1] > 256:
            img = thf.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False).detach()
            image_rec = thf.interpolate(image_rec, size=(256, 256), mode='bilinear', align_corners=False)
        if hasattr(self, 'noise') and self.noise.is_activated():
            image_rec_noised = self.noise(image_rec, self.global_step, p=0.9)
        else:
            image_rec_noised = image_rec
        pred = self.decoder(image_rec_noised)

        loss, loss_dict = self.loss_layer(img, image_rec, posterior, c, pred, self.global_step)
        bit_acc = loss_dict["bit_acc"]

        bit_acc_ = bit_acc.item()

        if (bit_acc_ == 1) and (not self.fixed_input) and self.noise.is_activated():
            self.loss_layer.activate_ramp(self.global_step)

        if (bit_acc_ > 0.99) and (not self.fixed_input):  # ramp up image loss at late training stage
            if hasattr(self, 'noise') and (not self.noise.is_activated()):
                self.noise.activate(self.global_step)

        if (bit_acc_ > 0.9) and self.fixed_input:  # execute only once
            print(f'[TRAINING]-achieved bit acc ({bit_acc_}), change origina image training.')
            self.fixed_input = ~self.fixed_input
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        loss_dict_no_ema = {f"val/{key}": val for key, val in loss_dict_no_ema.items() if key != 'img_lw'}
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {'val/' + key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def log_images(self, batch, fixed_input=False, **kwargs):
        log = dict()
        if fixed_input and self.fixed_img is not None:
            x, c, img, img_recon = self.fixed_x, self.fixed_control, self.fixed_img, self.fixed_input_recon
        else:
            x, c, img, img_recon = self.get_input(batch, return_first_stage=True)
        x, _ = self(x, img, c)
        image_out = self.decode_first_stage(x)
        if hasattr(self, 'noise') and self.noise.is_activated():
            img_noise = self.noise(image_out, self.global_step, p=1.0)
            log['noised'] = img_noise
        log['input'] = img
        log['output'] = image_out
        log['recon'] = img_recon
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer
