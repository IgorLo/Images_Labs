import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.image as mpimg


# Non Local Mean filter

def nlmfunc(i, j, fw, fh, nw, nh, image, sigma1, sigma2, nlmWFilter):
    imgmain = image[i - fh // 2:i + 1 + fh // 2, j - fw // 2:j + 1 + fw // 2, :]

    nlmFilter = 0
    for p in range(-(nh // 2), 1 + (nh // 2)):
        for q in range(-(nw // 2), 1 + (nw // 2)):
            imgneighbour = image[i + p - fh // 2: i + 1 + p + fh // 2, j + q - fw // 2:j + 1 + q + fw // 2, :]
            nlmIFilter = ((imgmain - imgneighbour) ** 2) / (2 * (sigma1 ** 2))
            nlmFilter += np.exp(-1 * nlmIFilter)

    nlmFilter = nlmFilter / np.sum(nlmFilter, axis=(0, 1))
    nlmFilter = nlmFilter * nlmWFilter
    nlmFilter = nlmFilter / np.sum(nlmFilter, axis=(0, 1))
    return np.sum(np.multiply(imgmain, nlmFilter), axis=(0, 1))


def gfunc(x, y, sigma):
    return (math.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))) / (2 * 3.14 * (sigma ** 2))


def gaussFilter(size, sigma):
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i, j] = gfunc(i - size[0] // 2, j - size[1] // 2, sigma)
    return out / np.sum(out)


def nlmFilterConv(image, fw, fh, nw, nh):
    col, row = image.shape
    sigma1 = 20
    sigma2 = 20
    nlmWFilter1 = 2 * 3.14 * sigma2 * sigma2 * gaussFilter((fw, fh), sigma2)
    if len(image.shape) < 3 or image.shape[2] == 1:
        nlmWFilter = np.resize(nlmWFilter1, (*nlmWFilter1.shape, 1))
    else:
        nlmWFilter = np.stack([nlmWFilter1, nlmWFilter1, nlmWFilter1], axis=2)

    out = np.zeros((col - 2 * fw + 1 - nw // 2, row - 2 * fh + 1 - nh // 2))
    for i in range(nh // 2, col - 2 * fh + 1 - nh // 2):
        for j in range(nw // 2, row - 2 * fw + 1 - nw // 2):
            out[i, j, :] = nlmfunc(i + fw - 1, j + fh - 1, fw, fh, nw, nh, image, sigma1, sigma2, nlmWFilter)

    out[0:nh // 2, :, :] = out[nh // 2, :, :]
    out[:, 0:nw // 2, :] = out[:, nw // 2, :, np.newaxis]
    if id == 1:
        return np.resize(out, (out.shape[0], out.shape[1])).astype(np.uint8)
    else:
        return out.astype(np.uint8)


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
    filename = "../images/sister.jpg"

    image = cv2.imread(filename, 0)
    image_noised = put_noise(image, 10)

    compare = cv2.hconcat([image, image_noised])

    nlm_image = nlmFilterConv(image_noised, 8, 8, 8, 8)
    cv2.imshow("NLM", nlm_image)
    cv2.waitKey(0)
