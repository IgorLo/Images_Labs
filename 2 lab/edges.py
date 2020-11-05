import cv2
import numpy as np
import math


def normalize2d(array):
    normalizedImg = np.zeros((len(array), len(array[0])))
    normalizedImg = cv2.normalize(array, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    return normalizedImg


def roberts_op(source, x, y):
    g_x = int(source[x + 1][y + 1]) - int(source[x][y])
    g_y = int(source[x][y + 1]) - int(source[x + 1][y])
    g = math.sqrt(g_x * g_x + g_y * g_y)
    return g


def prewitt_sobel_op(source, x, y, k):
    bottom_line = int(source[x - 1][y - 1]) + k * int(source[x][y - 1]) + int(source[x + 1][y - 1])
    top_line = int(source[x - 1][y + 1]) + k * int(source[x][y + 1]) + int(source[x + 1][y + 1])
    left_line = int(source[x - 1][y - 1]) + k * int(source[x - 1][y]) + int(source[x - 1][y + 1])
    right_line = int(source[x + 1][y - 1]) + k * int(source[x + 1][y]) + int(source[x + 1][y + 1])
    g_x = bottom_line - top_line
    g_y = right_line - left_line
    g = math.sqrt(g_x * g_x + g_y * g_y)
    return g


def expand(array):
    expanded = expandLine(array)
    result = []
    for i in range(0, len(expanded)):
        result.append(expandLine(expanded[i]))
    return np.uint8(result)


def expandLine(line):
    expanded = np.concatenate(([line[0]], line), axis=0)
    expanded = np.concatenate((expanded, [line[-1]]), axis=0)
    return expanded


if __name__ == '__main__':
    filename = "../images/dog4.jpg"

    image = cv2.imread(filename, 0)
    robertsImage = cv2.imread(filename, 0)
    prewittImage = cv2.imread(filename, 0)
    sobelImage = cv2.imread(filename, 0)

    image = expand(image)

    for i in range(1, len(image) - 1):
        for j in range(1, len(image[i]) - 1):
            robertsImage[i - 1][j - 1] = roberts_op(image, i, j)
            prewittImage[i - 1][j - 1] = prewitt_sobel_op(image, i, j, 1)
            sobelImage[i - 1][j - 1] = prewitt_sobel_op(image, i, j, 2)

    print(max(map(max, robertsImage)))
    print(min(map(min, robertsImage)))
    # print(max(map(max, prewittImage)))
    # print(min(map(min, prewittImage)))
    # print(max(map(max, sobelImage)))
    # print(min(map(min, sobelImage)))

    robertsImage = normalize2d(robertsImage)
    prewittImage = normalize2d(prewittImage)
    sobelImage = normalize2d(sobelImage)

    print(max(map(max, robertsImage)))
    print(min(map(min, robertsImage)))
    # print(max(map(max, prewittImage)))
    # print(min(map(min, prewittImage)))
    # print(max(map(max, sobelImage)))
    # print(min(map(min, sobelImage)))

    cv2.imshow("Expanded Original", image)
    cv2.imshow("Roberts", robertsImage)
    cv2.imshow("Prewitt", prewittImage)
    cv2.imshow("Sobel", sobelImage)
    cv2.waitKey(0)
