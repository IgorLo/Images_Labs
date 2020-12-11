import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def calc_diff(image1, image2):
    rows, cols = image1.shape
    square = rows * cols
    return np.sum(np.abs(image1 - image2)) / square


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def dnorm(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def calc_pixel(image, x, y, filter_size, sigma_r, sigma_s):
    fsum = 0
    wp = 0
    for i in range(filter_size):
        for j in range(filter_size):
            neighbour_x = int(x - (filter_size / 2 - i))
            neighbour_y = int(y - (filter_size / 2 - j))
            gr = dnorm(image[neighbour_x, neighbour_y] - image[x, y], sigma_r)
            gs = dnorm(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gr * gs
            fsum += image[neighbour_x, neighbour_y] * w
            wp += w
    return int(round(fsum / wp))


def bilateral_filter(image, sigma_r, sigma_s):
    output = np.zeros(image.shape)
    rows, cols = image.shape
    filter_size = int(sigma_r * 2 + 1) // 2 * 2 + 1
    if filter_size < 3:
        filter_size = 3
    pad = filter_size // 2
    padded_image = np.zeros((rows + (2 * pad), cols + (2 * pad)))
    padded_image[pad:padded_image.shape[0] - pad, pad:padded_image.shape[1] - pad] = image
    for i in range(rows):
        for j in range(cols):
            output[i, j] = calc_pixel(padded_image, i + pad, j + pad, filter_size, sigma_r, sigma_s)
            output[output > 255] = 255
            output[output < 0] = 0
    return output


def find_best_bilateral_sigma(image, noised_image, plot_filename='./img/bilateral_plot.jpg',
                              sigma_r_range=5, sigma_s_range=5, sigma_r_step=0.2,
                              sigma_s_step=1, sigma_r_start=0, sigma_s_start=0):
    best_image = bilateral_filter(noised_image, 1, 2)
    best_diff, best_sigma_r, best_sigma_s = calc_diff(image, best_image), 1, 2
    fig, axes = plt.subplots(nrows=sigma_r_range, ncols=sigma_s_range, figsize=(20, 20))
    for i in range(0, sigma_r_range):
        sigma_r = sigma_r_start + (i + 1) * sigma_r_step
        plt.setp(axes[i, 0], ylabel=('sigma_r = %.2f' % sigma_r))
        for j in range(0, sigma_s_range):
            sigma_s = sigma_s_start + (j+1) * sigma_s_step
            axes[0][j].set_title('sigma_s = %.2f' % sigma_s)
            print('Bilateral filter is processing %s%%'
                  % int((i * sigma_s_range + j + 1)/(sigma_r_range*sigma_s_range) * 100))
            blur_image = bilateral_filter(noised_image, sigma_r, sigma_s)
            diff = calc_diff(image, blur_image)
            axes[i][j].set_xlabel('Diff value: %.6f' % diff)
            axes[i][j].set_yticklabels([])
            axes[i][j].set_xticklabels([])
            axes[i][j].imshow(blur_image, cmap='gray')
            if diff < best_diff:
                best_diff, best_image, best_sigma_r, best_sigma_s = diff, blur_image, sigma_r, sigma_s
    plt.savefig(plot_filename, dpi=100)
    return best_image, best_sigma_r, best_sigma_s, best_diff


def put_noise(image, sigma):
    gauss = np.random.normal(0, sigma, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss
    cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy = noisy.astype(np.uint8)
    return normalize2d(noisy, 0, 255)


def normalize2d(array, bottom, top):
    normalizedImg = np.zeros((len(array), len(array[0])))
    normalizedImg = cv2.normalize(array, normalizedImg, bottom, top, cv2.NORM_MINMAX)
    return normalizedImg


if __name__ == '__main__':
    # input filename without extension
    filename_base = '../images/sister'

    image_in_color = cv2.imread('%s%s' % (filename_base, '.jpg'))
    image = cv2.imread('%s%s' % (filename_base, '.jpg'), cv2.IMREAD_GRAYSCALE).astype(np.float)
    rows, cols = image.shape
    square = rows * cols

    noised_image = put_noise(image, 10)
    cv2.imwrite('%s%s' % (filename_base, '_noised.jpg'), noised_image)

    bilateral_image, bilateral_sigma_r, bilateral_sigma_s, bilateral_diff = find_best_bilateral_sigma(
        image, noised_image, plot_filename='%s%s' % (filename_base, '2_bilateral_plot.jpg'),
        sigma_r_range=4, sigma_s_range=4,
        sigma_r_step=1, sigma_s_step=1,
        sigma_r_start=1, sigma_s_start=1)
    print('The best bilateral sigma_r: %s and sigma_s: %s' % (bilateral_sigma_r, bilateral_sigma_s))
    print('The best bilateral diff: %s' % bilateral_diff)
    cv2.imwrite('%s%s' % (filename_base, '_bilateral_filter.jpg'), bilateral_image.astype(np.uint8))
    vis_bilateral = cv2.hconcat([image, noised_image, bilateral_image])
    cv2.imwrite('%s%s' % (filename_base, '_bilateral_compare.jpg'), vis_bilateral)
