# sepwater
Beyond Direct Embedding: Secure Separable Latent Space Watermarking for Anti-Screen-Shooting
# Algorithm framework diagram：
<img width="500" height="300" alt="屏幕截图 2025-11-15 151240" src="https://github.com/user-attachments/assets/2d4b4bab-aa1c-404a-9247-24df8f905ee1" />

# Comparison images：
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/d54fec2a-ee32-4534-875d-8f060d0cd897" />


# Dependencies and Installation
Ubuntu >= 18.04  
CUDA >= 11.0  
Python 3.8.20  

\# git clone this repository
git clone[(https://github.com/CVhnu/sepwater.git) ](https://github.com/CVhnu/sepwater.git)

\# create new anaconda env
conda create -n sepwater python=3.8


\# Install dependencies
pip install -r requirement.txt


# Prepare
1. Download the pretrained checkpoints.
2. Preparing data for training

\# Creating a dataset table
You need to split the dataset into training and testing sets, and create CSV files for both sets, placing them in their respective directories.And download the corresponding weight using the link we provided.


The released model can be downloaded at
[(Download)]()

# run
python embed.py --config ../../../Embed_Ad_Extract.yaml --weight ../../../last.ckpt --secret xxy --cover ../../../###original-image --output ../../../###output-path
python extract.py --config ../../../Embed_Ad_Extract.yaml --weight ../../../last.ckpt --cover ../../../###output-path
CUDA_VISIBLE_DEVICES=1 python train.py --config ../../../train.yaml --secret_len ****** --max_image_weight_ratio *** --batch_size ***    ####you can change *****











