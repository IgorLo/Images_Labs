import cv2
import numpy as np
from scipy import signal


#preparing the image
def my_Normalize(img):
    # convert image into float matrix range of [0,1]
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    output = (img.astype('float') - min_val) / (max_val - min_val)
    return output

# №1 smooth and compute the derivatives of smoothed images
def my_DerivativesOfGaussian(img, sigma):

    smooth_img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)

    # Sobel kernel
    Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # Gaussian kernel
    #  Gaussian filter with a standard deviation of 3
    halfSize = 3 * sigma
    maskSize = 2 * halfSize + 1
    mat = np.ones((maskSize, maskSize)) / (float)(2 * np.pi * (sigma ** 2))
    xyRange = np.arange(-halfSize, halfSize + 1)
    xx, yy = np.meshgrid(xyRange, xyRange)
    x2y2 = (xx ** 2 + yy ** 2)
    exp_part = np.exp(-(x2y2 / (2.0 * (sigma ** 2))))
    gSig = mat * exp_part

    # the derivative of Gaussian kernels
    gx = signal.convolve2d(Sx, gSig)
    gy = signal.convolve2d(Sy, gSig)

    # применение ядер к Ix & Iy
    Ix = cv2.filter2D(img, -1, gx)
    Iy = cv2.filter2D(img, -1, gy)

    cv2.imshow('smooth', smooth_img)
    normGr = (Ix) + (Iy)
    cv2.imshow('Ix', my_Normalize(Ix))
    cv2.imshow('Iy', my_Normalize(Iy))
    return Ix, Iy


# №2 compute the magnitude and orientaion of the gradient image
def my_MagAndOrientation(Ix, Iy, t_low):
    # compute magnitude
    mag = np.sqrt(Ix ** 2 + Iy ** 2)

    # normalize magnitude image
    normMag = my_Normalize(mag)

    # compute orientation of gradient
    orient = np.arctan2(Iy, Ix)

    # round elements of orient
    orientRows = orient.shape[0]
    orientCols = orient.shape[1]


    for i in range(0, orientRows):
        for j in range(0, orientCols):
            if normMag[i, j] > t_low:
                # case 0 (0 degrees)
                # (if > 0  but < 22.5) or (if > 157.5 but < 180)
                if (orient[i, j] > (- np.pi / 8) and orient[i, j] <= (np.pi / 8)):
                    orient[i, j] = 0
                elif (orient[i, j] > (7 * np.pi / 8) and orient[i, j] <= np.pi):
                    orient[i, j] = 0
                    # elif (orient[i, j] >= -np.pi and orient[i, j] < (-7 * np.pi / 8)):
                    #    orient[i, j] = 0
                # case 1 (45 degrees)
                # if > 22.5 but < 67.5
                elif (orient[i, j] > (np.pi / 8) and orient[i, j] <= (3 * np.pi / 8)):
                    orient[i, j] = 3
                elif (orient[i, j] >= (-7 * np.pi / 8) and orient[i, j] < (-5 * np.pi / 8)):
                    orient[i, j] = 3
                # case 2 (90 degrees)
                # if > 67.5 but < 112.5
                elif (orient[i, j] > (3 * np.pi / 8) and orient[i, j] <= (5 * np.pi / 8)):
                    orient[i, j] = 2
                elif (orient[i, j] >= (-5 * np.pi / 4) and orient[i, j] < (-3 * np.pi / 8)):
                    orient[i, j] = 2
                # case 3 (135 degrees)
                # if > 112.5 but < 157.5
                elif (orient[i, j] > (5 * np.pi / 8) and orient[i, j] <= (7 * np.pi / 8)):
                    orient[i, j] = 1
                elif (orient[i, j] >= (-3 * np.pi / 8) and orient[i, j] < (-np.pi / 8)):
                    orient[i, j] = 1



    # show normalized magnitude image
    cv2.imshow('magnitude', normMag)

    return normMag, orient


# №4 non-maximal suppression along the gradient direction
def my_Non_Maximum_Suppression(mag, orient, t_low):
    mag_thin = np.zeros(mag.shape)
    for i in range(mag.shape[0] - 1):
        for j in range(mag.shape[1] - 1):
            if mag[i][j] < t_low:
                continue
            if orient[i][j] == 0:
                if mag[i][j] > mag[i][j - 1] and mag[i][j] >= mag[i][j + 1]:
                    mag_thin[i][j] = mag[i][j]
            if orient[i][j] == 1:
                if mag[i][j] > mag[i - 1][j + 1] and mag[i][j] >= mag[i + 1][j - 1]:
                    mag_thin[i][j] = mag[i][j]
            if orient[i][j] == 2:
                if mag[i][j] > mag[i - 1][j] and mag[i][j] >= mag[i + 1][j]:
                    mag_thin[i][j] = mag[i][j]
            if orient[i][j] == 3:
                if mag[i][j] > mag[i - 1][j - 1] and mag[i][j] >= mag[i + 1][j + 1]:
                    mag_thin[i][j] = mag[i][j]

    cv2.imshow('mag_thin', mag_thin)
    return mag_thin


# №5 linking using hysteresis thresholding
def my_Linking(mag_thin, orient, tLow, tHigh):
    result_binary = np.zeros(mag_thin.shape)

    # forward scan
    for i in range(0, mag_thin.shape[0] - 1):  # rows
        for j in range(0, mag_thin.shape[1] - 1):  # columns
            if mag_thin[i][j] >= tHigh:
                if mag_thin[i][j + 1] >= tLow:  # right
                    mag_thin[i][j + 1] = tHigh
                if mag_thin[i + 1][j + 1] >= tLow:  # bottom right
                    mag_thin[i + 1][j + 1] = tHigh
                if mag_thin[i + 1][j] >= tLow:  # bottom
                    mag_thin[i + 1][j] = tHigh
                if mag_thin[i + 1][j - 1] >= tLow:  # bottom left
                    mag_thin[i + 1][j - 1] = tHigh

    # backwards scan
    for i in range(mag_thin.shape[0] - 1, 0, -1):  # rows
        for j in range(mag_thin.shape[1] - 1, 0, -1):  # columns
            if mag_thin[i][j] >= tHigh:
                if mag_thin[i][j - 1] > tLow:  # left
                    mag_thin[i][j - 1] = tHigh
                if mag_thin[i - 1][j - 1]:  # top left
                    mag_thin[i - 1][j - 1] = tHigh
                if mag_thin[i - 1][j] > tLow:  # top
                    mag_thin[i - 1][j] = tHigh
                if mag_thin[i - 1][j + 1] > tLow:  # top right
                    mag_thin[i - 1][j + 1] = tHigh

    # fill in result_binary
    for i in range(0, mag_thin.shape[0] - 1):  # rows
        for j in range(0, mag_thin.shape[1] - 1):  # columns
            if mag_thin[i][j] >= tHigh:
                result_binary[i][j] = 1  # set to 1 for >= tHigh

    return result_binary


tLow = 0.01
tHigh = 0.2

img = cv2.imread("1.png",0)
print(img)
imgNorm = my_Normalize(img)
cv2.imshow('gray', imgNorm)
Ix, Iy = my_DerivativesOfGaussian(imgNorm, 1)
mag, orient = my_MagAndOrientation(Ix, Iy, tLow)
mag_thin = my_Non_Maximum_Suppression(mag, orient, tLow)
result = my_Linking(mag_thin, orient, tLow, tHigh)
canny = cv2.Canny(img, 100, 200)
cv2.imshow('my_canny', result)
cv2.imshow('opencv_canny', canny)

cv2.waitKey(0)

