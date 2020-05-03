import cv2 as cv
import numpy as np
import math
channel = 3
def Warpping(images, focal_lengths):
    #warpped_images = np.zeros(images.shape, dtype=np.uint8)
    warpped_images = []
    for i in range(len(images)):
        f = focal_lengths[i]
        s = focal_lengths[i]
        warpped_images.append(cylindricalWarpping(images[i], f, s))
        cv.imwrite("image_cyl" + str(i) + ".jpg", warpped_images[i])
    warpped_images = np.asarray(warpped_images)
    return warpped_images
def cylindricalWarpping(img, f, s):
    height, width= img.shape[:2]
    print("w, h" , width, height)
    res = np.zeros([height, width, channel],dtype=np.uint8 )
    x_0 = width/2
    y_0 = height/2
    for y in range(height):
      for x in range(width):
        theta = math.atan((x - x_0) / f)
        h = (y - y_0) / math.sqrt((x - x_0) ** 2 + f ** 2) * s
        #print(int(x_0 + s * theta), int(y_0 + h))
        res[int(y_0 + h), int(x_0 + s * theta), :] = img[y, x, :]
        
        
    idx = np.argwhere(np.all(res[..., :] == [0,0,0], axis=0))
    res = np.delete(res, idx, axis=1)
    return res
