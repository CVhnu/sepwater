import os, torch
import argparse
import numpy as np
from torchvision import transforms
from latent_diffmodel.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from anOther_funC.notes import welcome_message
from anOther_funC.ecc import ECC

def extract(args):
    print(welcome_message())
    os.makedirs(args.cover, exist_ok=True)
    input_dir = args.cover
    os.makedirs(output_dir, exist_ok=True)

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

    i, j = 0, 0
    with torch.no_grad():
        for file_name in os.listdir(input_dir):
            j += 1
            file_path = os.path.join(input_dir, file_name)
            cover_org = Image.open(file_path).convert('RGB')
            cover = tform(cover_org).unsqueeze(0).cuda()

            secret_pred = (model.decoder(cover) > 0).cpu().numpy()
            secret_decoded = ecc.decode_text(secret_pred)[0]
            print(f'Recovered secret: {secret_decoded}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", required=True, help="Path to config file.")
    parser.add_argument('-w', "--weight", required=True, help="Path to checkpoint file.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size.")
    parser.add_argument("--cover", required=True, help="Cover or stego image folder path")
    args = parser.parse_args()

    extract(args)
