import numpy as np

def imgs_concatenate(full_img, img, shift, r_shift):
	width = full_img.shape[1]+abs(shift[1])
	if r_shift+shift[0]<0:
		height = full_img.shape[0] + abs(r_shift+shift[0])
	elif shift[0]>0:
		height = full_img.shape[0] + shift[0]
	else:
		height = full_img.shape[0]
	
	img1 = np.zeros((height, width, 3))
	img2 = np.zeros((height, width, 3))
	
	#let the left image be img1 and right image be image2
	if shift[1] < 0 and r_shift+shift[0]> 0:
		img2[:full_img.shape[0],abs(shift[1]):,:] = full_img
		img1[r_shift+shift[0]:r_shift+shift[0]+img.shape[0], :img.shape[1],:] = img
		
	elif shift[1] < 0 and r_shift+shift[0] <= 0:
		img2[abs(r_shift+shift[0]):,abs(shift[1]):,:] = full_img
		img1[:img.shape[0], :img.shape[1],:] = img
		
	elif shift[1]>=0 and r_shift+shift[0]> 0:
		img1[:full_img.shape[0], :full_img.shape[1],:] = full_img
		img2[r_shift+shift[0]:r_shift+shift[0]+img.shape[0], -img.shape[1]:,:] = img
	else:
		img1[abs(r_shift+shift[0]):, :full_img.shape[1]] = full_img
		img2[:img.shape[0],-img.shape[1]: ] = img
		
	new_img = np.zeros((height, width, 3))
	if shift[1]<0:
		blend_left = abs(shift[1])
		blend_right = img.shape[1]
	else:
		blend_left = full_img.shape[1] + shift[1] - img.shape[1]
		blend_right = full_img.shape[1] 
		
	new_img[:, :blend_left,:] = img1[:, :blend_left,:]
	new_img[:, blend_right:,:] = img2[:, blend_right:,:]
	
	for i in range(blend_left, blend_right):
		new_img[:,i,:] = (img1[:,i,:]*(blend_right-i) + img2[:,i,:]*(i-blend_left))/(blend_right-blend_left)
	
	return new_img, 0 if r_shift+shift[0]<0 else r_shift+shift[0]
	
	
def global_warping(img, drift):
	new_height = img.shape[0]- abs(drift)
	new_img = np.zeros((new_height, img.shape[1], 3))
	# do global warping
	
	a = drift/img.shape[1]
	if a > 0:
		for c in range(img.shape[1]):
			new_img[:,c,:] = img[int(c*a):int(c*a)+new_height,c,:]
	else:
		for c in range(img.shape[1]):
			new_img[:,c,:] = img[abs(drift)+int(c*a):abs(drift)+int(c*a)+new_height,c,:] 
	return new_img
