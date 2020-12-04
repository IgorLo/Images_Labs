import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def calculateGaussian(x, y, sigma):
    left = 1 / (sigma * sigma * 2 * math.pi)
    power = -1 * (x * x + y * y)/(2 * sigma * sigma)
    right = math.pow(math.e, power)
    # return round(left * right)
    return left * right


def buildGaussianKernel(sigma, side):
    if side % 2 == 0:
        return []
    kernel = np.zeros((side, side))
    shift = math.floor(side / 2)
    for i in range(0, side):
        for j in range(0, side):
            kernel[i][j] = calculateGaussian(i - shift, j - shift, sigma)
    return kernel * (1 / kernel[0][0])


def normalize2d(array, bottom, top):
    normalizedImg = np.zeros((len(array), len(array[0])))
    normalizedImg = cv2.normalize(array, normalizedImg, bottom, top, cv2.NORM_MINMAX)
    return normalizedImg


def expand(array):
    expanded = expandLine(array)
    result = []
    for i in range(0, len(expanded)):
        result.append(expandLine(expanded[i]))
    return result


def expandLine(line):
    # print(line)
    expanded = np.concatenate(([line[0]], line), axis=0)
    expanded = np.concatenate((expanded, [line[-1]]), axis=0)
    # print(expanded)
    return expanded


def put_noise(image, sigma):
    gauss = np.random.normal(0, sigma, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss
    cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy = noisy.astype(np.uint8)
    return normalize2d(noisy, 0, 255)


def applyCoreIndividual(expanded, result, size, core, x, y):
    value = 0
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            value += math.floor(expanded[i + x][j + y] * core[i + size][j + size])
    result[x - size][y - size] = value / (len(core) * len(core))
    return


def applyCore(noised, result, core):
    expansionSize = math.floor(len(core)/2)
    print(len(noised))
    expanded = noised
    for i in range(0, expansionSize):
        expanded = expand(expanded)
    for i in range(1, len(noised) - 1):
        print(i/len(result) * 100)
        for j in range(1, len(noised[i]) - 1):
            applyCoreIndividual(expanded, result, expansionSize, core, i + expansionSize, j + expansionSize)
    pass


if __name__ == '__main__':
    filename = "../images/dog10.jpg"

    image = cv2.imread(filename, 0)
    filtered = cv2.imread(filename, 0)
    image_noised = put_noise(image, 10)

    applyCore(image_noised, filtered, buildGaussianKernel(1, 5))
    # filtered = normalize2d(filtered, 0, 255)

    cv2.imshow("original", image)
    cv2.imshow("noised", image_noised)
    cv2.imshow("filtered", filtered)
    cv2.waitKey(0)
