import os
from . import utils
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from anOther_funC.ctnoisely import RandomImagenetC
from PIL import Image


class TransformNet(nn.Module):
    def __init__(self, rnd_bri=0.3, rnd_hue=0.1, do_jpeg=False, jpeg_quality=50, rnd_noise=0.02, rnd_sat=1.0, rnd_trans=0.1,contrast=[0.5, 1.5], ramp=1000, imagenetc_level=0) -> None:
        super().__init__()
        self.rnd_bri = rnd_bri
        self.rnd_hue = rnd_hue
        self.jpeg_quality = jpeg_quality
        self.rnd_noise = rnd_noise
        self.rnd_sat = rnd_sat
        self.rnd_trans = rnd_trans
        self.contrast_low, self.contrast_high = contrast
        self.do_jpeg = do_jpeg
        self.ramp = ramp
        self.register_buffer('step0', torch.tensor(0))  # large number
        if imagenetc_level > 0:
            self.imagenetc = ImagenetCTransform(max_severity=imagenetc_level)

    def activate(self, global_step):
        if self.step0 == 0:
            print(f'[TRAINING] Activating TransformNet at step {global_step}')
            self.step0 = torch.tensor(global_step)

    def is_activated(self):
        return self.step0 > 0

    def forward(self, x, global_step, p=0.9):
        # x: [batch_size, 3, H, W] in range [-1, 1]
        if torch.rand(1)[0] >= p:
            return x
        if hasattr(self, 'imagenetc') and torch.rand(1)[0] < 0.5:
            x = self.imagenetc(x)
            return x
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        batch_size, sh, device = x.shape[0], x.size(), x.device
        ramp_fn = lambda ramp: np.min([(global_step-self.step0.cpu().item()) / ramp, 1.])

        rnd_bri = ramp_fn(self.ramp) * self.rnd_bri
        rnd_hue = ramp_fn(self.ramp) * self.rnd_hue
        rnd_brightness = utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size).to(device)  # [batch_size, 3, 1, 1]
        rnd_noise = torch.rand(1)[0] * ramp_fn(self.ramp) * self.rnd_noise

        contrast_low = 1. - (1. - self.contrast_low) * ramp_fn(self.ramp)
        contrast_high = 1. + (self.contrast_high - 1.) * ramp_fn(self.ramp)
        contrast_params = [contrast_low, contrast_high]

        rnd_sat = torch.rand(1)[0] * ramp_fn(self.ramp) * self.rnd_sat

        # blur
        N_blur = 7
        f = utils.random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                    wmin_line=3).to(device)
        x = F.conv2d(x, f, bias=None, padding=int((N_blur - 1) / 2))

        # noise
        noise = torch.normal(mean=0, std=rnd_noise, size=x.size(), dtype=torch.float32).to(device)
        x = x + noise
        x = torch.clamp(x, 0, 1)

        # contrast & brightness
        contrast_scale = torch.Tensor(x.size()[0]).uniform_(contrast_params[0], contrast_params[1])
        contrast_scale = contrast_scale.reshape(x.size()[0], 1, 1, 1).to(device)
        x = x * contrast_scale
        x = x + rnd_brightness
        x = torch.clamp(x, 0, 1)

        # saturation
        sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1).to(device)
        encoded_image_lum = torch.mean(x * sat_weight, dim=1).unsqueeze_(1)
        x = (1 - rnd_sat) * x + rnd_sat * encoded_image_lum

        # jpeg
        x = x.reshape(sh)
        if self.do_jpeg:
            jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(self.ramp) * (100. - self.jpeg_quality)
            x = utils.jpeg_compress_decompress(x, rounding=utils.round_only_at_0, quality=jpeg_quality)
        x = x * 2 - 1  # [0, 1] -> [-1, 1]
        return x


class ImagenetCTransform(nn.Module):
    def __init__(self, max_severity=5) -> None:
        super().__init__()
        self.max_severity = max_severity
        self.tform = RandomImagenetC(max_severity=max_severity, phase='train')

    def forward(self, x):#处理PIL图像类型
        # x: [batch_size, 3, H, W] in range [-1, 1]
        img0 = x.detach().cpu().numpy()
        img = img0 * 127.5 + 127.5  # [-1, 1] -> [0, 255]
        img = img.transpose(0, 2, 3, 1).astype(np.uint8)
        img = [Image.fromarray(i) for i in img]
        img = [self.tform(i) for i in img]
        img = np.array([np.array(i) for i in img], dtype=np.float32)
        img = img.transpose(0, 3, 1, 2) / 127.5 - 1.  # [0, 255] -> [-1, 1]
        residual = torch.from_numpy(img - img0).to(x.device)
        x = x + residual
        return x
