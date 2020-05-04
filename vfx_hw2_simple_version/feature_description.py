import numpy as np
import cv2 as cv
from math import pi, cos, sin, sqrt, floor
from scipy import interpolate
def interp(image, x_img, y_img):
    if(floor(x_img) - 1 >= 0 and floor(y_img) - 1 >= 0):
        x = [floor(x_img) - 1, floor(x_img), floor(x_img)+1]
        y = [floor(y_img) - 1, floor(y_img), floor(y_img)+1]
        z = image[x[0]:x[2]+1, y[0]: y[2]+1]
        f = interpolate.RectBivariateSpline(x, y, z, kx = 2, ky= 2)
        return f(x_img, y_img)
    else:
        x = [floor(x_img), floor(x_img)+1]
        y = [floor(y_img), floor(y_img)+1]
        z = image[x[0]:x[1]+1, y[0]: y[1]+1]
        f = interpolate.RectBivariateSpline(x, y, z, kx = 1, ky= 1)
        return f(x_img, y_img)
def SIFT_description_implement(image, key_point):
    print("assigning orientation...")
    #compute gaussian blurred image
    L = cv.GaussianBlur(image, (3, 3), 1.5)

    Dx = cv.Scharr(L, cv.CV_64F, 1, 0)
    Dy = cv.Scharr(L, cv.CV_64F, 0, 1)

    magnitude = np.sqrt(Dx ** 2 + Dy ** 2)
    angle = np.arctan2(Dy, Dx)
    for x, y in np.argwhere(angle == np.pi):
        angle[x][y]  = -np.pi
    #print("angle", angle)
    print("magnitude", magnitude)
    #cv.imwrite('magnitude.png', magnitude)
    num_bins = 36
    hist_step = 2*pi/num_bins
    bins = []
    for i in range(num_bins):
        bins.append(-pi + hist_step * i)
    window_size = 9
    hf_size = 4
    #np.digitize(angle, bins, right=False)
    key_point_pos = []
    major_orientation = []
    #image = image.astype('uint8')
    #img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for kp in key_point:
        if (kp[0] - hf_size) < 0 or (kp[0] + hf_size) >= image.shape[0]:
            continue
        if (kp[1] - hf_size) < 0 or (kp[1] + hf_size) >= image.shape[1]:
            continue
        currentWindow = angle[(kp[0] - hf_size):(kp[0] + hf_size + 1), (kp[1] - hf_size): (kp[1] + hf_size + 1)]
        mag = magnitude[(kp[0] - hf_size):(kp[0] + hf_size + 1), (kp[1] - hf_size): (kp[1] + hf_size + 1)]
        weightedMag = cv.GaussianBlur(mag, (window_size, window_size), sigmaX = 1.5)

        hist = np.digitize(currentWindow, bins, right=False)

        orient_bin = np.zeros([36])
        for i in range(window_size):
            for j in range(window_size):
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
        #img[kp[0]][kp[1]]=[0,255,0] 
    #cv.imwrite('check3.png', img)

    print("Local image descriptor")

    num_theta_bins = 8
    hist_step = 2 * pi / num_theta_bins 
    window_size = 16
    hf_window_size = 8
    bins = []
    for i in range(num_theta_bins):
        bins.append(-pi + hist_step * i)

    image = image.astype('uint8')
    img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    #create 16 * 16 window
    x = np.arange((0 - hf_window_size - 0.5), (0 + hf_window_size + 0.5), 1)
    y = np.arange((0 - hf_window_size - 0.5), (0 + hf_window_size + 0.5), 1)
    Xq, Yq = np.meshgrid(x, y)

    #for 16*16 grid
    sixteen_x = np.arange(0.5, 16.5, 1)
    sixteen_y = np.arange(0.5, 16.5, 1)
    Xs, Ys = np.meshgrid(sixteen_x, sixteen_y)
    Xs = np.reshape(Xs, (-1, 1))
    Ys = np.reshape(Ys, (-1, 1)) 
    sixteen_pos = np.hstack((Xs, Ys))

    feature_descriptors = []
    for i in range(len(major_orientation)):
        #create a rotated 16 * 16 window
        orientation = major_orientation[i]
        rotation_matrix = np.array([[float(cos(orientation)), -float(sin(orientation))],[float(sin(orientation)), float(cos(orientation))]])
        X = np.reshape(Xq, (-1, 1))
        Y = np.reshape(Yq, (-1, 1)) 

        pos = np.hstack((X, Y))
        if key_point_pos[i][0] - 13.5 < 0 or key_point_pos[i][0] + 13.5 >= image.shape[0]:
            continue
        if key_point_pos[i][1] - 13.5 < 0 or key_point_pos[i][1] + 13.5 >= image.shape[1]:
            continue
        #print("kp", key_point_pos[i][0], key_point_pos[i][1])
        cut_window = image[key_point_pos[i][0] - 8: key_point_pos[i][0] + 9, key_point_pos[i][1] - 8: key_point_pos[i][1] + 9]
        #cv.imwrite('cut_window.png', cut_window)

        count = 0
        original = np.zeros([17, 17])
        for k in range(17):
            for q in range(17):
                x_img = pos[count][0] + key_point_pos[i][0]
                y_img = pos[count][1] + key_point_pos[i][1]
                original[q][k] = interp(L, x_img, y_img)
                count += 1
        #cv.imwrite('original.png', original)

        pos = np.matmul(pos, rotation_matrix)
        X = np.reshape(pos[:, 0], (-1, 1))
        Y = np.reshape(pos[:, 1], (-1, 1))
        X = X + key_point_pos[i][0]
        Y = Y + key_point_pos[i][1]
        pr = np.hstack((X, Y))
        window_17 = np.zeros([17, 17])
        count = 0
        for k in range(17):
            for q in range(17):
                x_img = pr[count][0]
                y_img = pr[count][1]
                window_17[q][k] = interp(L, x_img, y_img)
                count += 1 
        #print("orientation", orientation)
        #cv.imwrite('window_17.png', window_17)

        #make real 16*16 window
        window = np.zeros([16, 16])
        count = 0
        for k in range(16):
            for q in range(16):
                window[q][k] = interp(window_17, sixteen_pos[count][0], sixteen_pos[count][1])
                count += 1
        #cv.imwrite('window.png', window)

        Dx_win = cv.Scharr(window, cv.CV_64F, 1, 0)
        Dy_win = cv.Scharr(window, cv.CV_64F, 0, 1)
        magnitude_win = np.sqrt(Dx_win ** 2 + Dy_win ** 2)
        angle_win = np.arctan2(Dy_win, Dx_win)
        angle_win[np.argwhere(angle_win == np.pi)] = -np.pi
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
        feature = []
        for c in range(16):
            mag = cells_mag[c].ravel()
            ang = cells_angle[c].ravel()
            #print("angle", ang)
            #print("bin", bins)
            hist = np.digitize(ang, bins, right=False)
            hist = hist - 1
            #print("hist", hist)
            orient_bin = np.zeros([8])
            for h in range(len(hist)):
                orient_bin[hist[h]] += mag[h]
            feature.append(orient_bin)
        feature = np.array(feature)
        feature = feature.ravel()
        #normaliztion
        denominator = np.linalg.norm(feature)
        if (denominator != 0):
            feature_normalized = feature / denominator
        feature_normalized[np.argwhere(feature_normalized > 0.2)] = 0.2

        denominator = np.linalg.norm(feature_normalized)
        if (denominator != 0):
            feature_normalized = feature_normalized / denominator
        feature_descriptors.append([(key_point_pos[i][0], key_point_pos[i][1]), feature_normalized])
        img[key_point_pos[i][0]][key_point_pos[i][1]]=[0,255,0]
    cv.imwrite('check2.png', img)
    
    return feature_descriptors


def feature_description(gray_images, key_points):
    descriptor = []
    for i in range(len(gray_images)):
        print(i)
        kps = np.argwhere(key_points[i] == 1)
        descriptor.append(SIFT_description_implement(gray_images[i], kps))
    return descriptor