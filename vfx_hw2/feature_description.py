import numpy as np
import cv2 as cv
from math import pi, cos, sin
from scipy import interpolate
def SIFT_description_implement(image, key_point):
    print("assigning orientation...")
    #compute gaussian blurred image
    L = cv.GaussianBlur(image, (3, 3), 0)
    Dx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize = 5)
    Dy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize = 5)
    magnitude = np.sqrt(Dx ** 2 + Dy ** 2)
    angle = np.arctan2(Dy, Dx)
    cv.imwrite('magnitude.png', magnitude)
    num_bins = 36
    hist_step = 2*pi/num_bins
    bins = []
    for i in range(num_bins):
        bins.append(-pi + hist_step * i)
    #print(bins)
    window_size = 9
    #np.digitize(angle, bins, right=False)
    key_point_pos = []
    major_orientation = []

    for kp in key_point:
        if (kp[0] - window_size) < 0 or (kp[0] + window_size) >= image.shape[0]:
            continue
        if (kp[1] - window_size) < 0 or (kp[1] + window_size) >= image.shape[1]:
            continue
        currentWindow = angle[(kp[0] - window_size):(kp[0] + window_size + 1), (kp[1] - window_size): (kp[1] + window_size + 1)]
        mag = magnitude[(kp[0] - window_size):(kp[0] + window_size + 1), (kp[1] - window_size): (kp[1] + window_size + 1)]
        weightedMag = cv.GaussianBlur(mag, (window_size * 2 + 1, window_size * 2 + 1), sigmaX = 1.5)
        hist = np.digitize(currentWindow, bins, right=False)
        orient_bin = np.zeros([36])
        for i in range(window_size * 2 + 1):
            for j in range(window_size * 2 + 1):
                orient_bin[hist[i][j] - 1] += weightedMag[i][j]

        first_peak_val = max(orient_bin)
        first_peak_ind = np.argwhere(orient_bin == first_peak_val)[0][0]
        key_point_pos.append(kp)
        major_orientation.append((first_peak_ind + 0.5) * pi / 36)

        temp_list = orient_bin
        temp_list[first_peak_ind] = 0
        second_peak_val = max(temp_list)
        second_peak_ind = np.argwhere(orient_bin == second_peak_val)[0][0]
        if(second_peak_val > 0.8 * first_peak_val):
            key_point_pos.append(kp)
            major_orientation.append((second_peak_ind + 0.5) * pi / 36)

    print("Local image descriptor")
    num_theta_bins = 8
    hist_step = 2 * pi / num_theta_bins 
    window_size = 16
    hf_window_size = 8
    bins = []
    for i in range(num_theta_bins):
        bins.append(-pi + hist_step * i)
    #create 16 * 16 window
    x = np.arange((0 - hf_window_size), (0 + hf_window_size), 1)
    y = np.arange((0 - hf_window_size), (0 + hf_window_size), 1)
    Xq,Yq = np.meshgrid(x, y)
    image = image.astype('uint8')
    img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    feature_descriptors = []
    for i in range(len(major_orientation)):
        #create a rotated 16 * 16 window
        orientation = major_orientation[i]
        rotation_matrix = np.array([[float(cos(orientation)), -float(sin(orientation))],[ float(sin(orientation)), float(cos(orientation))]])
        X = np.reshape(Xq, (-1, 1))
        Y = np.reshape(Yq, (-1, 1)) 
        pos = np.hstack((X, Y))
        pos = np.matmul(pos, rotation_matrix)
        X = np.reshape(pos[:, 0], (-1, 1))
        Y = np.reshape(pos[:, 1], (-1, 1))
        X = X + key_point_pos[i][0]
        Y = Y + key_point_pos[i][1]
        img[key_point_pos[i][0]][key_point_pos[i][1]]=[0,255,0] 
        pr = np.hstack((X, Y))
        window = np.zeros([16, 16])
        count = 0
        flag = 0
        for k in range(16):
            for q in range(16):
                if int(pr[count][0]) < 0 or int(pr[count][0]) >= image.shape[0]:
                    flag = 1
                    break
                if int(pr[count][1]) < 0 or int(pr[count][1]) >= image.shape[1]:
                    flag = 1
                    break
                count += 1
        if(flag == 1):
            continue
        count = 0
        for k in range(16):
            for q in range(16):
                x_img = int(pr[count][0])
                y_img = int(pr[count][1])
                window[k][q] = image[x_img][y_img]
                count += 1
        Dx_win = cv.Sobel(window, cv.CV_64F, 1, 0, ksize = 5)
        Dy_win = cv.Sobel(window, cv.CV_64F, 0, 1, ksize = 5)
        magnitude_win = np.sqrt(Dx_win ** 2 + Dy_win ** 2)
        angle_win = np.arctan2(Dy_win, Dx_win)
        weightedMag = cv.GaussianBlur(magnitude_win, (window_size+1, window_size+1), sigmaX = 8)
        
        #make into cells
        c4_wM = np.hsplit(weightedMag, 4)
        c4_angle = np.hsplit(angle_win, 4)
        cells_mag = []
        cells_angle = []
        for c in range(4):
            c16_wM = np.vsplit(c4_wM[c], 4)
            c16_angle = np.vsplit(c4_angle[c], 4)
            for q in range(4):
                cells_mag.append(c16_wM[q])
                cells_angle.append(c16_angle[q])
        #vote in cells
        count = 0
        feature = []
        for c in range(16):
            mag = cells_mag[c].ravel()
            ang = cells_angle[c].ravel()
            #print("reshaped", ang)
            hist = np.digitize(ang, bins, right=False)
            hist = hist - 1
            #print("hist", hist)
            orient_bin = np.zeros([8])
            for h in range(len(hist)):
                orient_bin[hist[h]] += mag[h]
            feature.append(orient_bin)
        feature = np.array(feature)

        feature_descriptors.append([(key_point_pos[i][0], key_point_pos[i][1]), feature.ravel()])
    cv.imwrite('check2.png', img)
    return feature_descriptors


def feature_description(gray_images, key_points):
    descriptor = []
    for i in range(len(gray_images)):
        print(i)
        kps = np.argwhere(key_points[i] == 1)
        descriptor.append(SIFT_description_implement(gray_images[i], kps))
    return descriptor