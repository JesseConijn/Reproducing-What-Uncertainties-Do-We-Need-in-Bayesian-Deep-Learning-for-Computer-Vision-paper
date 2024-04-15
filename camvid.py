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
    class_dict_path = os.path.join(path, 'class_dict.csv')
    classes = pd.read_csv(class_dict_path)['name'].values
    rgb = pd.read_csv(class_dict_path)[['r', 'g', 'b']].values
    
    # assign ID to color codes
    color_id = {tuple(v):k for k,v in enumerate(rgb)}
    return classes, rgb, color_id

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

def get_labels_paths(path):
    # Get images path
    train_labels_path = os.path.join(path, 'train_labels')
    test_labels_path = os.path.join(path, 'test_labels')
    val_labels_path = os.path.join(path, 'val_labels')

    # Get individual image paths
    train_labels_paths = glob.glob(os.path.join(train_labels_path, '*.png'))
    test_labels_paths = glob.glob(os.path.join(test_labels_path, '*.png'))
    val_labels_paths = glob.glob(os.path.join(val_labels_path, '*.png'))
    return train_labels_paths, test_labels_paths, val_labels_paths

def resize_images(images, new_height, new_width):
    resized_images = [cv2.resize(img, (new_width, new_height)) for img in images]
    return np.array(resized_images).reshape(resized_images.shape[0], -1) 

def get_image(image_path):
    # Load the images from the batch
    image = cv2.imread(image_path)
    return np.array(image)

def normalize_images(images):
    images = images / 255.
    images -= 0.4
    images /= 0.3
    return images

def color_map_labels(label, color_id):
    result = np.zeros_like(label, dtype=np.uint8)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            try:
                result[i, j] = color_id[tuple(label[i, j])]
            except KeyError:
                result[i, j] = color_id[tuple(np.array([0, 0, 0]))]      
    return result

def save_preprocessed_data(output_path_images, output_path_labels, images, labels):
    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_labels, exist_ok=True)

    for i, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(output_path_images, f'image_{i}.png')
        label_path = os.path.join(output_path_labels, f'label_{i}.png')

        cv2.imwrite(image_path, image)
        cv2.imwrite(label_path, label)
        
def load_preprocessed_data(output_path_images, output_path_labels, images, labels):
    # Get images path
    train_labels_path = os.path.join(path, 'train_labels')
    test_labels_path = os.path.join(path, 'test_labels')
    val_labels_path = os.path.join(path, 'val_labels')

    # Get individual image paths
    train_labels_paths = glob.glob(os.path.join(train_labels_path, '*.png'))
    test_labels_paths = glob.glob(os.path.join(test_labels_path, '*.png'))
    val_labels_paths = glob.glob(os.path.join(val_labels_path, '*.png'))
    return train_labels_paths, test_labels_paths, val_labels_paths
    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_labels, exist_ok=True)

    for i, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(output_path_images, f'image_{i}.png')
        label_path = os.path.join(output_path_labels, f'label_{i}.png')

        cv2.imwrite(image_path, image)
        cv2.imwrite(label_path, label)
        
# Function for preprocessing images and writing output images as TFRecords files          
def preprocess(images_path, output_path, image_height, image_width):
    
    # Get color labels
    classes, rgb, color_id = load_colors_label(images_path)

    # Get image and label paths
    train_image_paths, test_image_paths, val_image_paths = get_image_paths(images_path)
    train_label_paths, test_label_paths, val_label_paths = get_labels_paths(images_path)
    
    # Get output paths 
    train_out_path          = os.path.join(output_path, f'camvid-{image_height}x{image_width}-train')
    test_out_path           = os.path.join(output_path, f'camvid-{image_height}x{image_width}-test')
    val_out_path            = os.path.join(output_path, f'camvid-{image_height}x{image_width}-val')
    train_out_path_labels   = os.path.join(output_path, f'camvid-{image_height}x{image_width}-train-labels')
    test_out_path_labels    = os.path.join(output_path, f'camvid-{image_height}x{image_width}-test-labels')
    val_out_path_labels     = os.path.join(output_path, f'camvid-{image_height}x{image_width}-val-labels')
    
    # Preprocess images and labels
    train_images    = normalize_images(resize_images([cv2.imread(img) for img in train_image_paths], image_height, image_width))
    test_images     = normalize_images(resize_images([cv2.imread(img) for img in test_image_paths], image_height, image_width))
    val_images      = normalize_images(resize_images([cv2.imread(img) for img in val_image_paths], image_height, image_width))

    train_labels_colors = resize_images([cv2.imread(img) for img in train_label_paths], image_height, image_width)
    test_labels_colors  = resize_images([cv2.imread(img) for img in test_label_paths], image_height, image_width)
    val_labels_colors   = resize_images([cv2.imread(img) for img in val_label_paths], image_height, image_width)

    train_labels    = [color_map_labels(label, color_id) for label in train_labels_colors]
    test_labels     = [color_map_labels(label, color_id) for label in test_labels_colors]
    val_labels      = [color_map_labels(label, color_id) for label in val_labels_colors]
    
    return train_images, test_images, val_images, train_labels, test_labels, val_labels
