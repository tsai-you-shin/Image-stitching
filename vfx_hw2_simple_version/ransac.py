from math import ceil, log
import random
import numpy as np
def RANSAC(best_matches):
	threshold = 100
	best_matches = np.array(best_matches)
	max_inliner = 0
	x1 = best_matches[:, 0]
	y1 = best_matches[:, 1]
	x2 = best_matches[:, 2]
	y2 = best_matches[:, 3]
	p = 0.5
	n = 2
	P = 0.99
	k = ceil( log(1 - P) / log(1- p**n))
	pairs_diff = np.array([x1 - x2, y1 - y2])
	diff = np.copy(pairs_diff)
	nMatch = len(best_matches)
	for i in range(k):
		index = np.random.randint(0, len(x1) - 1)
		shift = pairs_diff[:, index]
		diff = pairs_diff.transpose() - shift
		dist = (np.sum(diff ** 2, axis = 1) < threshold)
		inliner = np.sum(dist)
		if inliner > max_inliner:
			max_inliner = inliner
			best_shift = shift
	return best_shift
