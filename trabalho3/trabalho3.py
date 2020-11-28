#!/usr/bin/env python
import time
import argparse

import cv2
import numpy as np
from skimage import io
import scipy.ndimage as ndimage

# Creates a global reference:
μ = σ = imax = imin = med = None


def global_method(img, T=128):
    return img <= T


def bernsen(img):
    T = (imax + imin) / 2
    return img <= T


def niblack(img, k=-0.2):
    T = μ + k * σ
    return img <= T


def sauvola(img, k=0.5, R=128):
    T = μ * (1 + k * (σ / R - 1))
    return img <= T


def more(img, k=0.25, R=0.5, p=2, q=10):
    T = μ * (1 + p * np.exp((-q) * μ) + k * (σ / R - 1))
    return img <= T


def contrast(img):
    # Distance from local max and min:
    max_dist = imax - img
    min_dist = img - imin
    return max_dist <= min_dist  # if is closer to max (background)


def mean(img):
    return img <= μ


def median(img):
    return img <= med


meths = [global_method, bernsen, niblack, sauvola, more, contrast, mean, median]


def read_args():
    parser = argparse.ArgumentParser(description='Image halftoning.')
    parser.add_argument('-i', required=True,
                        help='The name of the input image.')
    parser.add_argument('-o', required=True,
                        help='The base name of the output images.')
    parser.add_argument('-n', required=False, default=5, type=int,
                        help='The size of the mask used in each local method.')
    parser.add_argument('-f', required=False, default=255, type=int,
                        help='The results are multiplied by f to improve contrast.')
    return parser.parse_args()


def img_read(args):
    return cv2.imread(args.i, cv2.IMREAD_GRAYSCALE).astype(float)


def save_img(out_img, out_name):
    io.imsave(out_name, out_img.astype(np.uint8), check_contrast=False)


if __name__ == '__main__':
    start = time.time()
    args = read_args()
    in_img = img_read(args)

    # These operations are really expensive, so they are done just once:
    size = (args.n, args.n)
    μ = ndimage.generic_filter(in_img, np.mean, size=size, mode='nearest')
    σ = ndimage.generic_filter(in_img, np.std, size=size, mode='nearest')
    imax = ndimage.generic_filter(in_img, np.max, size=size, mode='nearest')
    imin = ndimage.generic_filter(in_img, np.min, size=size, mode='nearest')
    med = ndimage.generic_filter(in_img, np.median, size=size, mode='nearest')

    for fun in meths:
        result = args.f * fun(in_img)
        save_img(result, '{}-{}.png'.format(args.o, fun.__name__))

    end = time.time()
    print('Time elapsed: {:.4f}s'.format(end - start))
