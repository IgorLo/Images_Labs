import cv2
import matplotlib.pyplot as plt
import math


def build_histogram(values, width):
    result_hist = [0] * width
    for i in values:
        result_hist[i] += 1
    return result_hist

def build_cumulative(hist):
    result_array = [0] * len(hist)
    result_array[0] = hist[0]
    for i in range(1, len(hist)):
        result_array[i] = result_array[i - 1] + hist[i]
    return [number / sum(hist) for number in result_array]


def trim_percent(values, percent):
    remove_size = math.floor(len(values) / 100 * percent)
    return values[remove_size:-remove_size]


def build_equalize_matrix(norm_cumulative):
    result_matrix = [0] * len(norm_cumulative)
    for i in range(len(result_matrix)):
        result_matrix[i] = min(math.floor(norm_cumulative[i] * len(norm_cumulative)), 255)
    return result_matrix


if __name__ == '__main__':
    image = cv2.imread("husky.jpg", 0)
    cv2.imshow("Original", image)

    all_values = []
    for i in image:
        for j in i:
            all_values.append(j)

    baseHistogram = build_histogram(all_values, 256)
    plt.plot(baseHistogram)
    plt.show()

    baseCumulative = build_cumulative(baseHistogram)
    plt.plot(baseCumulative)
    plt.show()

    transform_matrix = build_equalize_matrix(baseCumulative)
    for i in range(len(all_values)):
        all_values[i] = transform_matrix[all_values[i]]
    finalHistogram = build_histogram(all_values, 256)
    plt.plot(finalHistogram)
    plt.show()

    finalCumulative = build_cumulative(finalHistogram)
    plt.plot(finalCumulative)
    plt.show()

    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = transform_matrix[image[i][j]]

    cv2.imshow("Result", image)

    cv2.waitKey(0)
