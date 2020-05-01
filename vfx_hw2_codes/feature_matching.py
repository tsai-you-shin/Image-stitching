import annoy
import numpy as np
def match(descriptors):
	num_image = len(descriptors)
	f = 128
	trees = []
	for i in range(num_image):
		t = annoy.AnnoyIndex(f, "euclidean")
		t.build(500)
		nFeatures = len(descriptors[i])
		for j in range(nFeatures):
			t.add_item(j, descriptors[i][j][1])
		trees.append(t)
	best_matches = []
	for i in range(num_image):
		best_match = []
		nFeatures = len(descriptors[i])
		for n in range(nFeatures):
			best = np.array([-1] * num_image)
			fd = descriptors[i][n][1]
			#search for the best match for a feature in other images
			for j in range(num_image):
				#skip the same image
				if(i == j):
					continue
				#other point
				ind, dist = t.get_nns_by_vector(fd, 2, search_k=-1, include_distances = True)
				#print("ind, dist", ind, dist)
				if(dist[1] == 0 or dist[0]/ dist[1] < 0.8):
					best[j] = ind[0]
				best_match.append(best)
		best_matches.append(best_match)
	return best_matches
	printf("Feature matching complete...")