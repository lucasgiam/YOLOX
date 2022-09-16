import os
import random
import sys
from pathlib import Path
from shutil import copyfile

import numpy as np

root_path = "./datasets/VOCdevkit/"
if not os.path.exists(root_path):
    os.makedirs(root_path)

xmlfilepath = root_path + "VOC2012/Annotations/"
if not os.path.exists(xmlfilepath):
    os.makedirs(xmlfilepath)
imagefilepath = root_path + "VOC2012/JPEGImages/"

if not os.path.exists(imagefilepath):
    os.makedirs(imagefilepath)

original_annot_path = r"C:\Users\Lucas_Giam\PKD_PPE\YOLOX\datasets\pascal_voc\annotations"
original_image_path = r"C:\Users\Lucas_Giam\PKD_PPE\YOLOX\datasets\pascal_voc\images"

# Move annotations to annotations folder
for filename in os.listdir(original_annot_path):
    print(filename)
    if filename.endswith(".xml"):

        copyfile(
            f"{original_annot_path}/{filename}",
            f"{xmlfilepath}/{filename}",
        )

for filename in os.listdir(original_image_path):
    if filename.endswith(".jpg"):
        copyfile(
            f"{original_image_path}/{filename}",
            f"{imagefilepath}/{filename}",
        )


txtsavepath = root_path + "/VOC2012/ImageSets/Main"

# if not os.path.exists(root_path):
#    print("cannot find such directory: " + root_path)
#    exit()

if not os.path.exists(txtsavepath):

    os.makedirs(txtsavepath)


seed = 1992
print("seed: ", seed)
random.seed(seed)
np.random.seed(seed)

trainval_percent = 0.9
train_percent = 0.8
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size:", tv)
print("train size:", tr)

ftrainval = open(txtsavepath + "/trainval.txt", "w")
ftest = open(txtsavepath + "/test.txt", "w")
ftrain = open(txtsavepath + "/train.txt", "w")
fval = open(txtsavepath + "/val.txt", "w")

for i in list:
    name = total_xml[i][:-4] + "\n"
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
