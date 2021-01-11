import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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


def add_noise(image):
    row, col = image.shape
    mean = 0
    # var = 0.1
    sigma = 10 # var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    draw_image(gauss, 'out/' + "1", '/only_noise_img.jpg')
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def main_func():
    source_folder_number = 1
    img_extension = "jpeg"

    # чтение изображения
    img = cv2.imread('resources/' + str(source_folder_number) + '/resource.' + img_extension, cv2.IMREAD_GRAYSCALE).astype(np.int32)
    draw_image(img, 'out/' + str(source_folder_number), '/resource_img.' + img_extension)

    #наложение шума
    new_image = add_noise(img)
    draw_image(new_image, 'out/' + str(source_folder_number), '/noise_img.' + img_extension)


if __name__ == '__main__':
    main_func()