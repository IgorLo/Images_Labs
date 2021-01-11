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


def img_binarization(img, threshold_value=128):
    binarized_img = img.copy()
    img_height, img_width = img.shape

    for i in range(0, img_height):
        for j in range(0, img_width):
            if binarized_img[i][j] >= threshold_value:
                binarized_img[i][j] = 255
            else:
                binarized_img[i][j] = 0

    return binarized_img


# проверка большинства пикселей чёрного цвета
def is_black_main(img):
    img_height, img_width = img.shape
    number_of_black_pix = 0
    number_of_white_pix = 0

    for i in range(0, img_height):
        for j in range(0, img_width):
            if img[i][j] == 0:
                number_of_white_pix = number_of_white_pix + 1
            else:
                number_of_black_pix = number_of_black_pix + 1

    return number_of_black_pix >= number_of_white_pix


def get_mask(size):
    return np.zeros((size, size))


# эрозия
def erosion(img, mask):
    erosion_img = img.copy()
    img_height, img_width = img.shape
    mask_height, mask_width = mask.shape
    mask_half_height = int(round(mask_height / 2))
    mask_half_width = int(round(mask_width / 2))

    for i in range(mask_half_height - 1, int(img_height - mask_half_height + 1)):
        for j in range(mask_half_width - 1, img_width - mask_half_width + 1):
            if img[i][j] == mask[mask_half_width][mask_half_width]:
                change = False
                for ii in range(i - mask_half_height + 1, i + mask_half_height - 1):
                    for jj in range(j - mask_half_width + 1, j + mask_half_width - 1):
                        if not(img[ii][jj] == mask[mask_half_width][mask_half_width]):
                            change = True
                if change:
                    for ii in range(i - mask_half_height + 1, i + mask_half_height - 1):
                        for jj in range(j - mask_half_width + 1, j + mask_half_width - 1):
                            erosion_img[ii][jj] = 0 if mask[mask_half_width][mask_half_width] == 255 else 255
    return erosion_img


# наращивание / дилетация
def escalating(img, mask):
    escalating_img = img.copy()
    img_height, img_width = img.shape
    mask_height, mask_width = mask.shape
    mask_half_height = int(round(mask_height/2))
    mask_half_width = int(round(mask_width/2))

    for i in range(mask_half_height - 1, img_height - mask_half_height + 1):
        for j in range(mask_half_width - 1, img_width - mask_half_width + 1):
            if img[i][j] == mask[mask_half_width][mask_half_width]:
                for ii in range(i - mask_half_height + 1, i + mask_half_height - 1):
                    for jj in range(j - mask_half_width + 1, j + mask_half_width - 1):
                        escalating_img[ii][jj] = img[i][j]

    return escalating_img


# отсеивает объекты меньшие структурного элемента, избегая сильного уменьшения объектов
def morphological_discovery(img, mask):
    erosion_img = erosion(img, mask)
    escalating_img = escalating(erosion_img, mask)
    return escalating_img


# избавляет от малых дыр и щелей, но увеличивает контур объекта
def morphological_closure(img, mask):
    escalating_img = escalating(img, mask)
    erosion_img = erosion(escalating_img, mask)
    return erosion_img


def main_func():
    source_folder_number = 1
    img_extension = "jpg"
    mask_size = 5

    # чтение изображения
    img = cv2.imread('resources/' + str(source_folder_number) + '/source.' + img_extension, cv2.IMREAD_GRAYSCALE)
    draw_image(img, './out/' + str(source_folder_number) + '/' + str(mask_size), '/resource_img.' + img_extension)

    # бинаризация изображения
    binarized_img = img_binarization(img)
    draw_image(binarized_img, './out/' + str(source_folder_number) + '/' + str(mask_size) , '/binarized_img.' + img_extension)

    mask = get_mask(mask_size)

    escalating_img = escalating(binarized_img, mask)
    draw_image(escalating_img, './out/' + str(source_folder_number) + '/' + str(mask_size) , '/escalating_img.' + img_extension)
    is_black_main(escalating_img)

    erosion_img = erosion(binarized_img, mask)
    draw_image(erosion_img, './out/' + str(source_folder_number) + '/' + str(mask_size) , '/erosion_img.' + img_extension)

    # морфологическое открытие
    morphological_discovery_img = morphological_discovery(binarized_img, mask)
    draw_image(morphological_discovery_img, './out/' + str(source_folder_number) + '/' + str(mask_size),
               '/morphological_discovery_img.' + img_extension)

    # морфологическое закрытие
    morphological_closure_img = morphological_closure(binarized_img, mask)
    draw_image(morphological_closure_img, './out/' + str(source_folder_number) + '/' + str(mask_size),
               '/morphological_closure_img.' + img_extension)

    return


if __name__ == '__main__':
    main_func()