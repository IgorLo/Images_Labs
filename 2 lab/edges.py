import cv2
import matplotlib.pyplot as plt
import math


def roberts_op(source, x, y):
    g_x = int(source[x + 1][y + 1]) - int(source[x][y])
    g_y = int(source[x][y + 1]) - int(source[x + 1][y])
    g = math.sqrt(g_x * g_x + g_y * g_y)
    return min(255, max(g, 0))


def prewitt_op(source, x, y):
    bottom_line = int(source[x - 1][y - 1]) + int(source[x][y - 1]) + int(source[x + 1][y - 1])
    top_line = int(source[x - 1][y + 1]) + int(source[x][y + 1]) + int(source[x + 1][y + 1])
    left_line = int(source[x - 1][y - 1]) + int(source[x - 1][y]) + int(source[x - 1][y + 1])
    right_line = int(source[x + 1][y - 1]) + int(source[x + 1][y]) + int(source[x + 1][y + 1])
    g_x = bottom_line - top_line
    g_y = right_line - left_line
    g = math.sqrt(g_x * g_x + g_y * g_y)
    return min(255, max(g, 0))


def sobel_op(source, x, y):
    bottom_line = int(source[x - 1][y - 1]) + 2 * int(source[x][y - 1]) + int(source[x + 1][y - 1])
    top_line = int(source[x - 1][y + 1]) + 2 * int(source[x][y + 1]) + int(source[x + 1][y + 1])
    left_line = int(source[x - 1][y - 1]) + 2 * int(source[x - 1][y]) + int(source[x - 1][y + 1])
    right_line = int(source[x + 1][y - 1]) + 2 * int(source[x + 1][y]) + int(source[x + 1][y + 1])
    g_x = bottom_line - top_line
    g_y = right_line - left_line
    g = math.sqrt(g_x * g_x + g_y * g_y)
    return min(255, max(g, 0))


if __name__ == '__main__':
    filename = "dog4.jpg"

    image = cv2.imread(filename, 0)
    robertsImage = cv2.imread(filename, 0)
    prewittImage = cv2.imread(filename, 0)
    sobelImage = cv2.imread(filename, 0)

    for i in range(1, len(image) - 1):
        for j in range(1, len(image[i]) - 1):
            robertsImage[i][j] = roberts_op(image, i, j)
            prewittImage[i][j] = prewitt_op(image, i, j)
            sobelImage[i][j] = sobel_op(image, i, j)

    cv2.imshow("Result", image)
    cv2.imshow("Roberts", robertsImage)
    cv2.imshow("Prewitt", prewittImage)
    cv2.imshow("Sobel", sobelImage)
    cv2.waitKey(0)
