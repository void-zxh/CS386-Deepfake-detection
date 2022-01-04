import shutil
import os
import random
from tqdm import tqdm

path="./ResNet34/data/train/real"
original_path="./FaceBlending/X-Ray Data Generator/dataset/images/real_mask/"
files=[]
dst_train="./ResNet34/data/train_mask/real"
for file in os.listdir(path): 
    files.append(file)

for file in tqdm(files):
    shutil.copy(original_path+"/"+file, dst_train)
# print(val_list)
# print(len(val_list))
# print(train_list)
# print(len(train_list))