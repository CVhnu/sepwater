# sepwater
Beyond Direct Embedding: Secure Separable Latent Space Watermarking for Anti-Screen-Shooting



<img width="800" height="500" alt="屏幕截图 2025-11-15 151240" src="https://github.com/user-attachments/assets/2d4b4bab-aa1c-404a-9247-24df8f905ee1" />


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


\# Creating a dataset table:
You need to split the dataset into training and testing sets, and create CSV files for both sets, placing them in their respective directories.


# Get Started
1. Download the pretrained checkpoints.
2. Preparing data for training

\# Creating a dataset table
You need to split the dataset into training and testing sets, and create CSV files for both sets, placing them in their respective directories

# Quick test
python inference.py 


## The released model can be downloaded at
[(Download)](https://drive.google.com/drive/folders/1BrYoGgqyrY_NUlJ-U4qn7h3aNQXvn5uy?usp=drive_link)


# Citation
If you find our repo useful for your research, please cite us:


# License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first.

# Acknowledgement

Code is inspired by ([RIDCP](https://github.com/RQ-Wu/RIDCP_dehazing)) and ([BasicSR](https://github.com/XPixelGroup/BasicSR)) .

