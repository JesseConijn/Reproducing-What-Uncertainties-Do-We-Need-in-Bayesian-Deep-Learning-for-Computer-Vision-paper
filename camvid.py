import os
import glob
import sys
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import csv
import cv2
from PIL import Image
from torchvision.datasets.folder import is_image_file, default_loader

def load_colors_label(path):
    classes = pd.read_csv(path).values[:,0]
    rgb = np.genfromtxt(path, delimiter=',', skip_header=1)[:,1:]
    return classes, rgb

def get_image_paths(path):
    # Get images path
    train_images_path = os.path.join(path, 'train')
    test_images_path = os.path.join(path, 'test')
    val_images_path = os.path.join(path, 'val')

    # Get individual image paths
    train_image_paths = glob.glob(os.path.join(train_images_path, '*.png'))
    test_image_paths = glob.glob(os.path.join(test_images_path, '*.png'))
    val_image_paths = glob.glob(os.path.join(val_images_path, '*.png'))
    return train_image_paths, test_image_paths, val_image_paths

# def get_images(image_paths):
#     images = []
#     for image_path in image_paths:
#         image = np.array(Image.open(image_path).resize((224,224), Image.NEAREST))
#         images.append(image)
#     return images

def get_images(image_paths):
    # Load the images from the batch
    images = [cv2.imread(image_path) for image_path in image_paths]
    return np.array(images)

def normalize_images(images):
    images = images / 255.
    return images

def get_mean_pixel_value(images):
    # Compute the mean pixel value across all images
    mean_pixel_value = np.mean(images, axis=(0, 1, 2)) 
    return mean_pixel_value

def get_std_pixel_value(images):
    std_pixel_value = np.std(images, axis=(0, 1, 2))
    return std_pixel_value

labels_path = './CamVid/class_dict.csv'
images_path = './CamVid/'
# Get image labels
classes, rgb = load_colors_label(labels_path)
print(rgb)

# Get image paths
train_images_paths, test_images_paths, val_images_paths = get_image_paths(images_path)

# # Get images
train_images = get_images(train_images_paths)
test_images = get_images(test_images_paths)
val_images = get_images(val_images_paths)
# #print("train_images: ", train_images)
# print("train_images2: ", train_images2)

# images_path = 'Camvid/train'  # Replace with the path to your folder containing images
mean_pixel_value = get_mean_pixel_value(train_images)
print("Mean pixel value:", mean_pixel_value)

std_pixel_value = get_std_pixel_value(train_images)
print("Std pixel value:", std_pixel_value)

normalized_train_images = normalize_images(train_images)  
mean_pixel_value_normalized = get_mean_pixel_value(normalized_train_images)
print("Mean pixel value:", mean_pixel_value_normalized)