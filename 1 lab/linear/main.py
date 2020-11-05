import cv2
import matplotlib.pyplot as plt
import math


def build_histogram(values, width, left, right):
    result_hist = [0] * width
    for line in values:
        for element in line:
            if element < left:
                continue
            if element > right:
                continue
            result_hist[element] += 1
    return result_hist


def trim_percent(percent, original_hist):
    left = 0
    right = 255
    total_space = sum(original_hist)
    new_space = total_space
    while new_space / total_space > 1.0 - percent:
        if original_hist[left] > original_hist[right]:
            right -= 1
        else:
            left += 1
        new_space = sum(original_hist[left:right])
    return [left, right]


def build_change_matrix(width, a, b):
    c = 0
    d = width
    result_matrix = [0] * width
    # (i - a) * ((d - c) / (b - a)) + c
    new_range = d - c
    old_range = b - a
    range_multiplier = new_range / old_range
    for i in range(len(result_matrix)):
        new_color = (i - a) * range_multiplier + c
        result_matrix[i] = max(min(math.floor(new_color), 255), 0)
    return result_matrix


if __name__ == '__main__':
    image = cv2.imread("../../images/dog2.jpg", 0)
    cv2.imshow("Original", image)

    baseHistogram = build_histogram(image, 256, 0, 255)
    plt.plot(baseHistogram)
    plt.show()

    trimmed = trim_percent(0.05, baseHistogram)
    print(trimmed)
    trimmedHistogram = build_histogram(image, 256, trimmed[0], trimmed[1])
    plt.plot(trimmedHistogram)
    plt.show()

    transform_matrix = build_change_matrix(256, trimmed[0], trimmed[1])

    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = transform_matrix[image[i][j]]

    finalHistogram = build_histogram(image, 256, 0, 255)
    plt.plot(finalHistogram)
    plt.show()

    cv2.imshow("Result", image)

    cv2.waitKey(0)
