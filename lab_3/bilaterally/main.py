import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def dnorm(x, sigma):
    return (1.0 / (2 * math.pi * (sigma**2))) * math.exp(-(x**2) / (2*sigma**2))


def calc_pixel(image, x, y, filter_size, sigma_r, sigma_s):
    fsum = 0
    wp = 0
    for i in range(filter_size):
        for j in range(filter_size):
            neighbour_x = int(x - (filter_size / 2 - i))
            neighbour_y = int(y - (filter_size / 2 - j))
            gr = dnorm(image[neighbour_x, neighbour_y] - image[x,y], sigma_r)
            gs = dnorm(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gr * gs
            fsum += image[neighbour_x, neighbour_y] * w
            wp += w
    return int(round(fsum / wp))


def bilateral_filter(image, sigma_r, sigma_s):
    output = np.zeros(image.shape)
    rows, cols = image.shape
    filter_size = 3
    pad = filter_size //2
    padded_image = np.zeros((rows + (2*pad), cols + (2*pad)))
    padded_image[pad:padded_image.shape[0] - pad, pad:padded_image.shape[1] - pad] = image
    for i in range(rows):
        for j in range(cols):
            output[i,j] = calc_pixel(padded_image, i + pad, j + pad, filter_size, sigma_r, sigma_s)
            if output[i,j] > 255:
                output[i,j] = 255
            if output[i,j] < 0:
                output[i,j] = 0
    return output


def calc_diff(img1, img2):
    rows, cols = img1.shape
    square = rows * cols
    return np.sum(np.abs(img1.astype(np.float) - img2.astype(np.float))) / square


def draw_image(img, saving_folder, saving_name):
    plt.tick_params(labelsize=0, length=0)
    plt.imshow(img, cmap='gray')
    Path(saving_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(saving_folder + saving_name, bbox_inches='tight', pad_inches=0)
    plt.show()


def main_func():
    source_folder_number = 1
    img_extension = "jpeg"
    sigma = 10

    # чтение изображения
    img = cv2.imread('../noise/out/' + str(source_folder_number) + '/resource_img.' + img_extension,
                           cv2.IMREAD_GRAYSCALE)
    noise_img = cv2.imread('../noise/out/' + str(source_folder_number) + '/noise_img.' + img_extension,
                     cv2.IMREAD_GRAYSCALE)

    sigma_spatial = [2, 4, 8, 10, 16]
    sigma_intensity = [16, 24, 32, 64, 96]

    for sigma in sigma_spatial:
        for sigma_int in sigma_intensity:
            filtered_img = bilateral_filter(noise_img, sigma, sigma_int)
            print('sigma: ' + str(sigma) + '; sigma_int: ' + str(sigma_int) + '; diff: ' + str(calc_diff(img, filtered_img)))
            draw_image(filtered_img, 'out/11' + str(source_folder_number), '/bilateral_img_' + str(sigma) + '_'
                       + str(sigma_int) + '.' + img_extension)


if __name__ == '__main__':
    main_func()