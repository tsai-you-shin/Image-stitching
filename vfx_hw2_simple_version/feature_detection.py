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

def NMS(dst):
	minimum = np.min(dst)
	
	win = 5
	dst[-win:,:] = minimum
	dst[:win, :] = minimum
	dst[:,-win:] = minimum
	dst[:, :win] = minimum
	
	key_points = np.zeros(dst.shape)
	collect_points = 0
	r = dst.shape[0]
	c = dst.shape[1]
	
	while True:
		if collect_points >= 500:
			break
		chose_point = np.argmax(dst)
		max_r = chose_point//c
		max_c = chose_point%c
		collect_points+=1
		key_points[max_r, max_c] = 1
		dst[max_r-win:max_r+win+1 ,max_c-win: max_c+win+1] = minimum

	return key_points
	
	
	
def feature_detect_implement(img, gray_img, count):
	gray = np.float32(gray_img)
	dst1 = cv2.cornerHarris(gray, 2, 3, 0.05)
	dst2 = cornerHarris(gray_img, 3, 0.05)
	img2 = img.copy()
	img1 = img.copy()
	img1[dst1>0.01*dst1.max()]=[0,0,255]
	key_points = NMS(dst2)
	img2[key_points == 1]=[0,255,0]
	
	cv2.imwrite('my gaussian'+str(count)+'.png',img2)
	
	#key_points = np.zeros(gray.shape)
	#key_points[dst2>0.01*dst2.max()] = 1
	
	return key_points

def feature_detection(images, gray_imgs):
	Rs = []
	for i in range(len(images)):
		Rs.append(feature_detect_implement(images[i], gray_imgs[i], i))
	return Rs
	
	
