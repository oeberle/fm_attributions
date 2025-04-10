#!/bin/bash

pip install zennit
pip install captum
pip install sklearn


git clone https://github.com/hassanhajj910/prompt-me-a-dataset.git 

wget "https://git.tu-berlin.de/pchormai/drsa-demo/-/raw/main/tests/data/castle.jpg?ref_type=heads" -O castle.jpg


## Get Weights
cd prompt-me-a-dataset/dino_files/weights && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && cd ..

cd prompt-me-a-dataset/sam_files/weights &&  wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && cd ..


pip install git+https://github.com/facebookresearch/segment-anything.git

pip install groundingdino-py

# Get Data
wget https://tubcloud.tu-berlin.de/s/ZCr7YxqyBX598cY/download/1808_clavius_sphaera_1585.zip && unzip 1808_clavius_sphaera_1585.zip  && rm -rf __MACOSX/ && rm 1808_clavius_sphaera_1585.zip
