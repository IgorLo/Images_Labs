import cv2
import matplotlib.pyplot as plt
import math


def build_histogram(values, width):
    result_hist = [0] * width
    for i in values:
        result_hist[i] += 1
    return result_hist


def trim_percent(values, percent):
    remove_size = math.floor(len(values) / 100 * percent)
    return values[remove_size:-remove_size]


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
    image = cv2.imread("dog.jpg", 0)
    cv2.imshow("Original", image)

    all_values = []
    for i in image:
        for j in i:
            all_values.append(j)

    baseHistogram = build_histogram(all_values, 256)
    plt.plot(baseHistogram)
    plt.show()

    all_values.sort()
    trimmed = trim_percent(all_values, 5)
    trimmedHistogram = build_histogram(trimmed, 256)
    plt.plot(trimmedHistogram)
    plt.show()

    transform_matrix = build_change_matrix(256, min(trimmed), max(trimmed))
    print(transform_matrix)
    for i in range(len(trimmed)):
        trimmed[i] = transform_matrix[trimmed[i]]
    finalHistogram = build_histogram(trimmed, 256)
    plt.plot(finalHistogram)
    plt.show()

    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = transform_matrix[image[i][j]]

    cv2.imshow("Result", image)

    cv2.waitKey(0)
