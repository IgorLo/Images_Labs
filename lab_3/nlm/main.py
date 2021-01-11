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


# функция сглаживания нелокальными средними
def nlm(img, sigma, patch_big=7, patch_small=3):
    # разбиваем на больше и маленькое окно для обхода вокруг центра пикселя
    # создаем пэддинг для изображения
    pad = patch_big + patch_small

    img_ = np.pad(img, pad, mode="reflect")

    result_img = np.zeros((img.shape[0], img.shape[1]))
    h, w = img_.shape

    # коэф. для гауссова распределения(параметр разрброса весов)
    # задается вручную
    H_gauss = 25

    # проход по всему изображению
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):

            current_val = 0

            # получаем большое окно для сдвига по картинке
            startY = y - patch_big
            endY = y + patch_big

            startX = x - patch_big
            endX = x + patch_big

            # нормирующий делитель и максимальное значение веса
            Wp, maxweight = 0, 0
            # обойти по всем соседям пикселя окна и найти похожее значение, рассчитать вес
            # аккумулировать вес и добавить в текущую картинку
            for ypix in range(startY, endY):
                for xpix in range(startX, endX):
                    # создаем для текущего пикселя окна для рассчета gauss - L2 norm
                    window1 = img_[y - patch_small:y + patch_small, x - patch_small:x + patch_small].copy()
                    window2 = img_[ypix - patch_small:ypix + patch_small, xpix - patch_small:xpix + patch_small].copy()

                    # подсчет весов для текущего пикселя
                    weight = np.exp(-(np.sum((window1 - window2) ** 2) + 2 * (sigma ** 2)) / (
                                H_gauss ** 2))

                    # находим максимальное значение веса
                    if weight > maxweight:
                        maxweight = weight

                    # если текущий пиксель совпадает с нужным нам похожим пикселем вес будет равен максимальному
                    if (y == ypix) and (x == xpix):
                        weight = maxweight

                    Wp += weight
                    current_val += weight * img_[ypix, xpix]
            # обновляем изображения с учетом нового веса
            result_img[y - pad, x - pad] = current_val / Wp

    return result_img


def calc_diff(img1, img2):
    rows, cols = img1.shape
    square = rows * cols
    return np.sum(np.abs(img1.astype(np.float) - img2.astype(np.float))) / square


def main_func():
    source_folder_number = 1
    img_extension = "jpeg"
    sigma = 10

    # чтение изображения
    img = cv2.imread('../noise/out/' + str(source_folder_number) + '/resource_img.' + img_extension,
                           cv2.IMREAD_GRAYSCALE).astype(np.int32)
    noise_img = cv2.imread('../noise/out/' + str(source_folder_number) + '/noise_img.' + img_extension,
                           cv2.IMREAD_GRAYSCALE).astype(np.int32)
    draw_image(noise_img, 'out/' + str(source_folder_number), '/resource_img.' + img_extension)

    nlm_img = nlm(noise_img, sigma, 7, 3)
    print('sigma: ' + str(sigma) + '; diff: '+ str(calc_diff(img, nlm_img)))
    draw_image(nlm_img, 'out/' + str(source_folder_number), '/nlm_img_' + str(sigma) + '.' + img_extension)
    #draw_image(comp_img, 'out/' + str(source_folder_number), '/nlm_img_comparison_' + str(sigma) + '.' + img_extension)

    sigma = 20
    nlm_img = nlm(noise_img, sigma, 7, 3)
    print('sigma: ' + str(sigma) + '; diff: '+ str(calc_diff(img, nlm_img)))
    draw_image(nlm_img, 'out/' + str(source_folder_number), '/nlm_img_' + str(sigma) + '.' + img_extension)
    #draw_image(comp_img, 'out/' + str(source_folder_number), '/nlm_img_comparison_' + str(sigma) + '.' + img_extension)

    sigma = 30
    nlm_img = nlm(noise_img, sigma, 7, 3)
    print('sigma: ' + str(sigma) + '; diff: '+ str(calc_diff(img, nlm_img)))
    draw_image(nlm_img, 'out/' + str(source_folder_number), '/nlm_img_' + str(sigma) + '.' + img_extension)
    #draw_image(comp_img, 'out/' + str(source_folder_number), '/nlm_img_comparison_' + str(sigma) + '.' + img_extension)

    sigma = 40
    nlm_img = nlm(noise_img, sigma, 7, 3)
    print('sigma: ' + str(sigma) + '; diff: '+ str(calc_diff(img, nlm_img)))
    draw_image(nlm_img, 'out/' + str(source_folder_number), '/nlm_img_' + str(sigma) + '.' + img_extension)
    #draw_image(comp_img, 'out/' + str(source_folder_number), '/nlm_img_comparison_' + str(sigma) + '.' + img_extension)

    sigma = 60
    nlm_img = nlm(noise_img, sigma, 7, 3)
    print('sigma: ' + str(sigma) + '; diff: '+ str(calc_diff(img, nlm_img)))
    draw_image(nlm_img, 'out/' + str(source_folder_number), '/nlm_img_' + str(sigma) + '.' + img_extension)
    #draw_image(comp_img, 'out/' + str(source_folder_number), '/nlm_img_comparison_' + str(sigma) + '.' + img_extension)

    # sigma = 14
    # nlm_img = nlm(noise_img, sigma, 7, 3)
    # comp_img = abs(img - nlm_img)
    # draw_image(nlm_img, 'out/' + str(source_folder_number), '/nlm_img_' + str(sigma) + '.' + img_extension)
    # draw_image(comp_img, 'out/' + str(source_folder_number), '/nlm_img_comparison_' + str(sigma) + '.' + img_extension)
    #
    # sigma = 60
    # nlm_img = nlm(noise_img, sigma, 7, 3)
    # comp_img = abs(img - nlm_img)
    # draw_image(nlm_img, 'out/' + str(source_folder_number), '/nlm_img_' + str(sigma) + '.' + img_extension)
    # draw_image(comp_img, 'out/' + str(source_folder_number), '/nlm_img_comparison_' + str(sigma) + '.' + img_extension)


if __name__ == '__main__':
    main_func()
