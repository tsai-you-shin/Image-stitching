import cv2 
import numpy as np


def cornerHarris(gray_img, blocksize, k):
	# find the matrix M for each pixel
	# E(u,v) = [u,v] M [u,v]^T 
	
	deri_x = cv2.Sobel(gray_img,cv2.CV_64F, 1, 0, ksize = 3) #horizentail
	deri_y = cv2.Sobel(gray_img,cv2.CV_64F, 0, 1, ksize = 3) #vertical
	
	deri_xx = np.multiply(deri_x,deri_x)
	deri_yy = np.multiply(deri_y,deri_y)
	deri_xy = np.multiply(deri_x,deri_y)
	
	i_xx = cv2.GaussianBlur(deri_xx, (blocksize,blocksize), 0)
	i_yy = cv2.GaussianBlur(deri_yy, (blocksize,blocksize), 0)
	i_xy = cv2.GaussianBlur(deri_xy, (blocksize,blocksize), 0)

	detM = i_xx *i_yy - (i_xy**2)
	traceM = i_xx + i_yy
	R = detM - k*(traceM**2)
	
	return R

def feature_detect(gray_img):
	gray = np.float32(gray_img)
	dst1 = cv2.cornerHarris(gray,2,3,0.04)
	dst2 = cornerHarris(gray_img, 3, 0.04)

	dst1 = cv2.dilate(dst1,None)
	img2 = img.copy()
	img[dst1>0.01*dst1.max()]=[0,0,255]
	img2[dst2>0.01*dst2.max()]=[0,255,0]

	cv2.imwrite('buildin gaussian.png',img)
	cv2.imwrite('my gaussian.png',img2)


img = cv2.imread("./grail/grail00.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
feature_detect(gray)