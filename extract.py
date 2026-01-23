import os
import sys
import torch
import argparse
import shutil
import numpy as np
from torchvision import transforms
from Latent_Diffusion_Model.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from tools.ecc import ECC
from tools.helpers import welcome_message
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Image Steganography - Bit Accuracy Extraction")
    parser.add_argument('-c', "--config", required=True, help="Path to config file.")
    parser.add_argument('-w', "--weight", required=True, help="Path to checkpoint file.")
    parser.add_argument("--image_size", type=int, default=224, help="Height and width of square images.")
    parser.add_argument("--secret", required=True, help="The expected secret message for validation.")
    parser.add_argument("--cover", required=True, help="Path to folder containing stego images.")
    parser.add_argument("-o", "--output", default=None, help="Path to save correctly decoded images (optional).")
    args = parser.parse_args()
    input_dir = args.cover
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    config = OmegaConf.load(args.config).model
    secret_len = config.params.control_config.params.secret_len
    config.params.decoder_config.params.secret_len = secret_len

    model = instantiate_from_config(config)
    state_dict = torch.load(args.weight, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    misses, ignores = model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()


    target_text = args.secret
    target_bits = ''.join([format(ord(c), '08b') for c in target_text])
    bit_len = len(target_bits)

    print(f"Target Secret: {target_text}")
    print(f"Target Bits Length: {bit_len}")
    print("-" * 30)


    tform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


    global_total_bits = 0
    global_correct_bits = 0
    total_img_count = 0

    ecc = ECC()

    with torch.no_grad():
        files = sorted(os.listdir(input_dir))

        for file_name in files:
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            total_img_count += 1
            file_path = os.path.join(input_dir, file_name)

            try:
                cover_org = Image.open(file_path).convert('RGB')
                cover = tform(cover_org).unsqueeze(0).cuda()


                secret_pred = (model.decoder(cover) > 0).cpu().numpy()[0]
                pred_bits = ''.join([str(int(b)) for b in secret_pred[:bit_len]])


                img_correct_bits = 0
                for pb, tb in zip(pred_bits, target_bits):
                    if pb == tb:
                        img_correct_bits += 1

                img_acc = img_correct_bits / bit_len
                print(f"[{file_name}] Bit Acc: {img_acc:.2%}")

                global_correct_bits += img_correct_bits
                global_total_bits += bit_len

                if args.output:

                    if img_acc == 1.0:
                        out_path = os.path.join(args.output, file_name)
                        shutil.copy(file_path, out_path)

            except Exception as e:
                print(f"[{file_name}] Error: {e}")

    print("-" * 30)
    print("-------- Final Statistics --------")
    if global_total_bits > 0:
        avg_acc = global_correct_bits / global_total_bits
        print(f"Processed Images: {total_img_count}")
        print(f"Average Bit Acc : {avg_acc:.2%}")
    else:
        print("No images processed.")


if __name__ == '__main__':
    main()
