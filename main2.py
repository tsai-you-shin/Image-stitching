from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import math
import argparse
import os
from cylindrical_warping import *
from feature_detection import *
from feature_description import *
from feature_matching2 import *
from ransac import *
from image_concatenate import *


def loadExposureSeq(path):
    images = []
    focal_lengths = []
    with open(os.path.join(path, 'pano_test.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join(path, tokens[0])))
        focal_lengths.append(float(tokens[1]))
    return images, np.asarray(focal_lengths, dtype=np.float32)

def get_gray(images):
    gray_images = []
    for i in range(len(images)):
        gray_images.append(cv.cvtColor(images[i], cv2.COLOR_BGR2GRAY))
    return gray_images

parser = argparse.ArgumentParser(description='Code for VFX project 2.')
parser.add_argument('--input', type=str, help='Path to the directory that contains images and focal lengths.')
args = parser.parse_args()
if not args.input:
    parser.print_help()
    exit(0)

## load in the images
print("Loading images...")
images, focal_lengths = loadExposureSeq(args.input)
images = np.array(images, dtype=np.uint8)
print(images.shape[0])
warpped_images = np.zeros(images.shape)

print("Cylindrical Warpping...")
warpped_images = Warpping(images, focal_lengths)

print("creating grayscale images")
gray_images = get_gray(warpped_images)

print("Feature Detecting...")
key_points = feature_detection(warpped_images, gray_images)
print("Feature describing...")
try:
	descriptors = np.load('descriptors.npy',  allow_pickle=True)
except:
	descriptors = feature_description(gray_images, key_points)
	print(len(descriptors), len(descriptors[0]), len(descriptors[1]))
	descriptors = np.array(descriptors)
	np.save('descriptors', descriptors)

drift = 0
for i in range(len(images) - 1):
    print("Feature matching...")
    best_matches = match(descriptors[i], descriptors[i + 1])
    draw_match(gray_images[i], gray_images[i + 1], descriptors[i], descriptors[i+1], best_matches)

    print("Image matching...")
    best_shift = RANSAC(best_matches)
    print("best shift:",best_shift)
    
    if i==0:
    	full_img,r_shift = imgs_concatenate(warpped_images[i],warpped_images[i+1], best_shift, 0)
    else:
    	full_img, r_shift = imgs_concatenate(full_img, warpped_images[i+1], best_shift, r_shift)
    drift += best_shift[0]*np.sign(best_shift[1])

print("drift=", drift)
cv.imwrite('full_img.png', full_img)
full_img = global_warping(full_img, drift)    
cv.imwrite('full_img_warp.png', full_img)


