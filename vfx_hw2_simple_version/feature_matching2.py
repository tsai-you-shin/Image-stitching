import annoy
import numpy as np
import cv2 as cv
def draw_match( gray_images1, gray_images2, descriptors1, descriptors2, best_matches):
	img1 =  gray_images1.astype('uint8')
	img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
	img2 =  gray_images2.astype('uint8')
	img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
	for i in range(len(best_matches)):
			x1 = best_matches[i][0]
			y1 = best_matches[i][1]
			x2 = best_matches[i][2]
			y2 = best_matches[i][3]
			#print("best_matches", i, x1, y1, x2, y2)
			img1[x1][y1] = [0,0,255]
			img2[x2][y2] = [0,255,0]
	cv.imwrite('match1.png', img1)
	cv.imwrite('match2.png', img2)

def match(descriptor1, descriptor2):
	f = 128
	matches = []
	t = annoy.AnnoyIndex(f, "euclidean")
	t.build(1000)
	nFeatures = len(descriptor1)
	for j in range(nFeatures):
		t.add_item(j, descriptor1[j][1])

	nFeatures = len(descriptor2)
	for n in range(nFeatures):
		fd = descriptor2[n][1]
		#search for the best match for a feature in other images
		ind, dist = t.get_nns_by_vector(fd, 2, search_k=-1, include_distances = True)
		#print("ind, dist", ind, dist)
		if(dist[1] == 0 or dist[0]/ dist[1] < 0.8):
			matches.append([descriptor1[ind[0]][0][0], descriptor1[ind[0]][0][1], descriptor2[n][0][0], descriptor2[n][0][1]]) 

	t2 = annoy.AnnoyIndex(f, "euclidean")
	t2.build(1000)
	nFeatures = len(descriptor2)
	for j in range(nFeatures):
		t2.add_item(j, descriptor2[j][1])

	nFeatures = len(descriptor1)
	for n in range(nFeatures):
		fd = descriptor1[n][1]
		#search for the best match for a feature in other images
		ind2, dist2 = t2.get_nns_by_vector(fd, 2, search_k=-1, include_distances = True)
		if(dist2[1] == 0 or dist2[0]/dist2[1] < 0.8):
			matches.append([descriptor1[n][0][0], descriptor1[n][0][1], descriptor2[ind2[0]][0][0], descriptor2[ind2[0]][0][1]])
	print("Feature matching complete...")
	return matches
	