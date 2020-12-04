import cv2
import matplotlib.pyplot as plt
import math


def build_histogram(values, width):
    result_hist = [0] * width
    for i in values:
        for j in i:
            result_hist[j] += 1
    return result_hist


def build_cumulative(hist):
    result_array = [0] * len(hist)
    result_array[0] = hist[0]
    for i in range(1, len(hist)):
        result_array[i] = result_array[i - 1] + hist[i]
    return [number / sum(hist) for number in result_array]


def normalize(array):
    lowerbound = min(array)
    upperbound = max(array)
    shift = -lowerbound
    width = upperbound - lowerbound
    scale = 255 / width
    normalized_array = [((x + shift) * scale) for x in array]
    return normalized_array


def build_equalize_matrix(norm_cumulative):
    result_matrix = [0] * len(norm_cumulative)
    for i in range(len(result_matrix)):
        result_matrix[i] = norm_cumulative[i] * len(norm_cumulative)
    return normalize(result_matrix)
    # return result_matrix


if __name__ == '__main__':
    image = cv2.imread("../../images/husky.jpg", 0)
    cv2.imshow("Original", image)

    baseHistogram = build_histogram(image, 256)
    plt.plot(baseHistogram)
    plt.show()

    baseCumulative = build_cumulative(baseHistogram)
    plt.plot(baseCumulative)
    plt.show()

    transform_matrix = build_equalize_matrix(baseCumulative)

    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = transform_matrix[image[i][j]]

    finalHistogram = build_histogram(image, 256)
    plt.plot(finalHistogram)
    plt.show()

    finalCumulative = build_cumulative(finalHistogram)
    plt.plot(finalCumulative)
    plt.show()

    cv2.imshow("Result", image)

    cv2.waitKey(0)
