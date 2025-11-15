import os, sys, torch
import argparse
from pathlib import Path
import numpy as np
from torchvision import transforms
from latent_diffmodel.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from anOther_funC.notes import welcome_message
from anOther_funC.ecc import ECC
import time

def embed(args):
    print(welcome_message())
    os.makedirs(args.cover, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    input_dir = args.cover
    output_dir = args.output

    config = OmegaConf.load(args.config).model
    secret_len = config.params.control_config.params.secret_len
    config.params.decoder_config.params.secret_len = secret_len
    model = instantiate_from_config(config)

    state_dict = torch.load(args.weight, map_location=torch.device('cpu'))
    if 'global_step' in state_dict:
        print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print(f'Missed keys: {misses}\nIgnore keys: {ignores}')
    model = model.cuda()
    model.eval()

    tform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    ecc = ECC()
    secret = ecc.encode_text([args.secret])  # 1, 100
    secret = torch.from_numpy(secret).cuda().float()

    with torch.no_grad():
        start_time = time.time()
        for file_name in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file_name)
            save_path = os.path.join(output_dir, file_name)

            cover_org = Image.open(file_path).convert('RGB')
            w, h = cover_org.size
            cover = tform(cover_org).unsqueeze(0).cuda()

            z = model.encode_first_stage(cover)
            z_embed, _ = model(z, None, secret)

            z_embed_bits = z_embed.numel() * z_embed.element_size() * 8
            stego = model.decode_first_stage(z_embed)

            res = stego.clamp(-1, 1) - cover
            res = torch.nn.functional.interpolate(res, (h, w), mode='bilinear')
            res = res.permute(0, 2, 3, 1).cpu().numpy()

            stego_uint8 = np.clip(res[0] + np.array(cover_org) / 127.5 - 1., -1, 1) * 127.5 + 127.5
            stego_uint8 = stego_uint8.astype(np.uint8)

            Image.fromarray(stego_uint8).save(save_path)
            print(f'Stego saved to {save_path}')

        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} sec")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", required=True, help="Path to config file.")
    parser.add_argument('-w', "--weight", required=True, help="Path to checkpoint file.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size.")
    parser.add_argument("--secret", default='xxy', help="Secret message")
    parser.add_argument("--cover", required=True, help="Cover image folder path")
    parser.add_argument("-o", "--output", required=True, help="Output folder path")
    args = parser.parse_args()

    embed(args)
