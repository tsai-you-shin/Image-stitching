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
from feature_matching import *
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
    gray_images = np.zeros([images.shape[0], images.shape[1], images.shape[2]])
    for i in range(len(images)):
        gray_images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
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
descriptors = feature_description(gray_images, key_points)
print(len(descriptors), len(descriptors[0]), len(descriptors[1]))

print("Feature matching...")
best_matches = match(descriptors)
print(len(best_matches))
check = []
for i in range(2):
    img = gray_images[i].astype('uint8')
    print(img)
    check.append(cv.cvtColor(img, cv.COLOR_GRAY2BGR))

for i in range(2):
    for j in range(len(best_matches[i])):
        for p in range(2):
            if i == p: continue
            if best_matches[i][j][p] != -1:
                x = descriptors[i][j][0][0]
                y = descriptors[i][j][0][1]
                print("best_matches", i, x, y)
                if i == 0:
                    check[i][x][y] = [0,0,255]
                else:
                    check[i][x][y] = [0,255,0]
                x = descriptors[p][best_matches[i][j][p]][0][0]
                y = descriptors[p][best_matches[i][j][p]][0][1]
                print("to best_matches", p, x, y)
                if p == 0:
                    check[p][x][y] = [0,0,255]
                else:
                    check[p][x][y] = [0,255,0]
                
cv2.imwrite('match1.png', check[0])
cv2.imwrite('match2.png', check[1])
