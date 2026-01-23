import os
import sys
import torch
import argparse
import numpy as np
from torchvision import transforms
from Latent_Diffusion_Model.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from tools.ecc import ECC
from tools.helpers import welcome_message


def main():
    parser = argparse.ArgumentParser(description="Image Steganography - Embedding")
    parser.add_argument('-c', "--config", required=True, help="Path to config file.")
    parser.add_argument('-w', "--weight", required=True, help="Path to checkpoint file.")
    parser.add_argument("--image_size", type=int, default=224, help="Height and width of square images.")
    parser.add_argument("--secret", required=True, help="Secret message to hide.")
    parser.add_argument("--cover", required=True, help="Path to folder containing cover images.")
    parser.add_argument("-o", "--output", required=True, help="Path to save stego images.")
    args = parser.parse_args()

    print(welcome_message())

    # 创建文件夹
    os.makedirs(args.output, exist_ok=True)
    input_dir = args.cover
    output_dir = args.output

    # Load model
    print(f"Loading model from {args.config}...")
    config = OmegaConf.load(args.config).model
    secret_len = config.params.control_config.params.secret_len
    config.params.decoder_config.params.secret_len = secret_len

    model = instantiate_from_config(config)
    state_dict = torch.load(args.weight, map_location=torch.device('cpu'))

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print(f'Model Loaded. Missed keys: {len(misses)}, Ignore keys: {len(ignores)}')

    model = model.cuda()
    model.eval()

    # cover transform
    tform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # secret encoding
    ecc = ECC()
    secret = ecc.encode_text([args.secret])
    secret = torch.from_numpy(secret).cuda().float()

    print(f"Embedding secret: '{args.secret}' into images...")

    with torch.no_grad():
        for file_name in os.listdir(input_dir):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            file_path = os.path.join(input_dir, file_name)
            save_path = os.path.join(output_dir, file_name)

            cover_org = Image.open(file_path).convert('RGB')
            w, h = cover_org.size
            cover = tform(cover_org).unsqueeze(0).cuda()

            z = model.encode_first_stage(cover)
            z_embed, _ = model(z, None, secret)

            stego = model.decode_first_stage(z_embed)

            # Residual calculation to preserve quality
            res = stego.clamp(-1, 1) - cover
            res = torch.nn.functional.interpolate(res, (h, w), mode='bilinear')
            res = res.permute(0, 2, 3, 1).cpu().numpy()

            stego_uint8 = np.clip(res[0] + np.array(cover_org) / 127.5 - 1., -1, 1) * 127.5 + 127.5
            stego_uint8 = stego_uint8.astype(np.uint8)

            Image.fromarray(stego_uint8).save(save_path)
            print(f'Stego saved to {save_path}')


if __name__ == '__main__':
    main()
