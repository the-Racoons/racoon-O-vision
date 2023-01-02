import torchvision.transforms  as T
import cv2
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt

def processing(imgs_path,label_path,new_size=128):
    #
    path = imgs_path
    labels = pd.read_csv(label_path)
    #
    labels["xmin"] = labels["xmin"] * new_size / labels["width"]
    labels["xmax"] = labels["xmax"] * new_size / labels["width"]
    labels["ymin"] = labels["ymin"] * new_size / labels["height"]
    labels["ymax"] = labels["ymax"] * new_size / labels["height"]
    #
    X = []
    Y = labels[["xmin","ymin","xmax","ymax"]]
    #
    for file in labels['filename'] :
        orignal_image = cv2.imread(os.path.join(path,file))
        image = cv2.cvtColor(orignal_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image , (new_size,new_size))
        X.append(resized_image)

    return (X,Y)
