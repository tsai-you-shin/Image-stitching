import cv2
import numpy as np
import math
channel = 3
def cylindricalWarpping(img, f, s):
    height, width= img.shape[:2]
    print("w, h" , width, height)
    res = np.zeros([height, width, channel])
    x_0 = width/2
    y_0 = height/2
    for y in range(height):
      for x in range(width):
        theta = math.atan((x - x_0) / f)
        h = (y - y_0) / math.sqrt((x - x_0) ** 2 + f ** 2) * s
        #print(int(x_0 + s * theta), int(y_0 + h))
        res[round(y_0 + h), round(x_0 + s * theta), :] = img[y, x, :]
    return res

if __name__ == '__main__':
  img = cv2.imread("./parrington/prtn00.jpg")
  print("cylindricalWarpping...")
  f = 704.916
  s = f
  cyl_warpped_img = cylindricalWarpping(img, f, s)
  cv2.imwrite("image_cyl.png", cyl_warpped_img)
