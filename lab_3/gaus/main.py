import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from math import ceil


def draw_image(img, saving_folder, saving_name):
    plt.tick_params(labelsize=0, length=0)
    plt.imshow(img, cmap='gray')
    Path(saving_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(saving_folder + saving_name, bbox_inches='tight', pad_inches=0)
    plt.show()


def draw_graph(arr, x_label, y_label, saving_folder, saving_name):
    plt.bar(np.arange(len(arr)), arr)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    Path(saving_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(saving_folder + saving_name, bbox_inches='tight', pad_inches=0)
    plt.show()


def gauss_blur(img, kernel_size, sigma):
    # создание Гауссова фильтра
    gauss_kernel = cv2.getGaussianKernel(kernel_size ** 2, sigma, cv2.CV_32F).reshape(kernel_size, kernel_size)

    img_h, img_w = img.shape
    flt_h, flt_w = gauss_kernel.shape

    padding_h = ceil((flt_h - 1) / 2)  # округление до ближайшего большего числа
    padding_w = ceil((flt_w - 1) / 2)

    # создаем новое изображение увеличенное по всем сторонам на padding размер
    # заполняем его нулями
    padding = np.zeros((img_h + (2 * padding_h), img_w + (2 * padding_w)))
    # присваиваем исх. изображение внутрь padding
    padding[padding_h:padding.shape[0] - padding_h, padding_w:padding.shape[1] - padding_w] = img

    result = np.zeros(img.shape)
    # рассчет свертки ядра для изображения
    for i in range(img_h):
        for j in range(img_w):
            result[i, j] = np.sum(gauss_kernel * padding[i:i + flt_h, j:j + flt_w])

    return color_renormalize(result)


# ренормализация интенсивности изображения
def color_renormalize(img):
    re_norm = img.copy()
    re_norm *= 255.0 / re_norm.max()
    re_norm = np.uint8(re_norm)
    return re_norm


def calc_diff(img1, img2):
    rows, cols = img1.shape
    square = rows * cols
    return np.sum(np.abs(img1.astype(np.float) - img2.astype(np.float))) / square


def main_func():
    source_folder_number = 1
    img_extension = "jpeg"
    sigma = 10
    kernel_size = 3

    # чтение изображения
    img = cv2.imread('../noise/out/' + str(source_folder_number) + '/resource_img.' + img_extension,
                           cv2.IMREAD_GRAYSCALE).astype(np.int32)
    noise_img = cv2.imread('../noise/out/' + str(source_folder_number) + '/noise_img.' + img_extension,
                           cv2.IMREAD_GRAYSCALE).astype(np.int32)
    draw_image(noise_img, 'out/' + str(source_folder_number), '/resource_img.' + img_extension)

    gauss_img = gauss_blur(noise_img, kernel_size, sigma).astype(np.int32)
    print('sigma: ' + str(sigma) + '; diff: '+ str(calc_diff(img, gauss_img.astype(np.int32))))
    draw_image(gauss_img, 'out/' + str(source_folder_number), '/gauss_img_sigma_' + str(sigma) + '.' + img_extension)

    sigma = 2
    gauss_img = gauss_blur(noise_img, kernel_size, sigma)
    print('sigma: ' + str(sigma) + '; diff: ' + str(calc_diff(img, gauss_img.astype(np.int32))))
    draw_image(gauss_img, 'out/' + str(source_folder_number), '/gauss_img_sigma_' + str(sigma) + '.' + img_extension)

    sigma = 4
    gauss_img = gauss_blur(noise_img, kernel_size, sigma)
    print('sigma: ' + str(sigma) + '; diff: ' + str(calc_diff(img, gauss_img.astype(np.int32))))
    draw_image(gauss_img, 'out/' + str(source_folder_number), '/gauss_img_sigma_' + str(sigma) + '.' + img_extension)

    sigma = 6
    gauss_img = gauss_blur(noise_img, kernel_size, sigma)
    print('sigma: ' + str(sigma) + '; diff: ' + str(calc_diff(img, gauss_img.astype(np.int32))))
    draw_image(gauss_img, 'out/' + str(source_folder_number), '/gauss_img_sigma_' + str(sigma) + '.' + img_extension)

    sigma = 8
    gauss_img = gauss_blur(noise_img, kernel_size, sigma)
    print('sigma: ' + str(sigma) + '; diff: ' + str(calc_diff(img, gauss_img.astype(np.int32))))
    draw_image(gauss_img, 'out/' + str(source_folder_number), '/gauss_img_sigma_' + str(sigma) + '.' + img_extension)

    sigma = 12
    gauss_img = gauss_blur(noise_img, kernel_size, sigma)
    print('sigma: ' + str(sigma) + '; diff: ' + str(calc_diff(img, gauss_img.astype(np.int32))))
    draw_image(gauss_img, 'out/' + str(source_folder_number), '/gauss_img_sigma_' + str(sigma) + '.' + img_extension)

    sigma = 14
    gauss_img = gauss_blur(noise_img, kernel_size, sigma)
    print('sigma: ' + str(sigma) + '; diff: ' + str(calc_diff(img, gauss_img.astype(np.int32))))
    draw_image(gauss_img, 'out/' + str(source_folder_number), '/gauss_img_sigma_' + str(sigma) + '.' + img_extension)

    sigma = 16
    gauss_img = gauss_blur(noise_img, kernel_size, sigma)
    print('sigma: ' + str(sigma) + '; diff: ' + str(calc_diff(img, gauss_img.astype(np.int32))))
    draw_image(gauss_img, 'out/' + str(source_folder_number), '/gauss_img_sigma_' + str(sigma) + '.' + img_extension)


if __name__ == '__main__':
    main_func()
