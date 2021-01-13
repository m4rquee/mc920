#!/usr/bin/env python
import time
import argparse
from math import floor
from functools import partial

import numpy as np
from skimage import io
from skimage.util import img_as_ubyte

BLANK_VALUE = 0


def read_args():
    parser = argparse.ArgumentParser(description='Image rotation/scaling.')
    parser.add_argument('-i', required=True,
                        help='The name of the input image.')
    parser.add_argument('-o', required=True,
                        help='The name of the output image.')
    parser.add_argument('-a', required=False, type=float,
                        help='Rotation angle measured counterclockwise (degrees).')
    parser.add_argument('-e', required=False, type=float,
                        help='Scaling factor.')
    parser.add_argument('-d', required=False, type=int, nargs=2,
                        help='Dimension of the output image in pixels.')
    parser.add_argument('-m', required=False, type=int, default=0, choices=range(4),
                        help='Interpolation method to use.')
    ret = parser.parse_args()

    if all(v is None for v in [ret.a, ret.e, ret.d]):
        parser.error('No action requested! Add -a, -e or -d.')
    return ret


def img_read(in_name):
    return img_as_ubyte(io.imread(in_name, as_gray=True))


def save_img(out_img, out_name):
    io.imsave(out_name, out_img, check_contrast=False)


# Creates a rotation matrix around a center with negative angles (indirect):
def rot_mat(theta, x_center, y_center):
    theta = theta * np.pi / 180  # converts to radians

    # Translation matrizes used to move the center of rotation:
    # note that the order is switched because the mapping is indirect
    mov_mat = np.array(((1, 0, -x_center), (0, 1, -y_center), (0, 0, 1)))
    mov_mat_prime = np.array(((1, 0, x_center), (0, 1, y_center), (0, 0, 1)))

    # Rotation matrix around the origin:
    cos, sin = np.cos(-theta), np.sin(-theta)
    rot = np.array(((cos, -sin, 0), (sin, cos, 0), (0, 0, 1)))

    return mov_mat_prime @ rot @ mov_mat  # compose the matrizes


# Creates a scaling matrix with inverse factors (indirect):
def scl_mat(x_factor, y_factor):
    return np.array(((1 / x_factor, 0, 0), (0, 1 / y_factor, 0), (0, 0, 1)))


def closest(f, w, h, x, y):  # the closest value in f to the (x, y) position
    x = round(x)
    y = round(y)
    if 0 <= x < w and 0 <= y < h:
        return f[x][y]
    return BLANK_VALUE


def bilinear(f, w, h, x, y):  # bilinear interpolation
    xf, yf = floor(x), floor(y)  # gets the upper left corner
    if 0 <= xf < w - 1 and 0 <= yf < h - 1:  # border values are ignored
        dx, dy = x - xf, y - yf
        return (1 - dx) * (1 - dy) * f[xf][yf] + dx * (1 - dy) * f[xf + 1][yf] + \
               (1 - dx) * dy * f[xf][yf + 1] + dx * dy * f[xf + 1][yf + 1]
    return BLANK_VALUE


P = lambda t: max(0, t)
def R(s):
    return (P(s + 2) ** 3 - 4 * P(s + 1) ** 3 + 6 * P(s) ** 3
            - 4 * P(s - 1) ** 3) / 6


def bicubic(f, w, h, x, y):  # bicubic interpolation
    xf, yf = floor(x), floor(y)  # gets the upper left corner
    if 1 <= xf < w - 2 and 1 <= yf < h - 2:  # border values are ignored
        dx, dy = x - xf, y - yf
        ret = 0
        for m in range(-1, 3):
            for n in range(-1, 3):
                ret += f[xf + m][yf + n] * R(m - dx) * R(dy - n)
        return ret
    return BLANK_VALUE


def L(f, dx, x, y, n):
    return (-dx * (dx - 1) * (dx - 2) * f[x - 1][y + n - 2] / 3
            + (dx + 1) * (dx - 1) * (dx - 2) * f[x][y + n - 2]
            - dx * (dx + 1) * (dx - 2) * f[x + 1][y + n - 2]
            + dx * (dx + 1) * (dx - 1) * f[x + 2][y + n - 2] / 3) / 2


def lagrange(f, w, h, x, y):  # Lagrange polynomials interpolation
    xf, yf = floor(x), floor(y)  # gets the upper left corner
    if 1 <= xf < w - 2 and 1 <= yf < h - 2:  # border values are ignored
        dx, dy = x - xf, y - yf
        L_curried = partial(L, f, dx, xf, yf)
        return (-dy * (dy - 1) * (dy - 2) * L_curried(1) / 3
                + (dy + 1) * (dy - 1) * (dy - 2) * L_curried(2)
                - dy * (dy + 1) * (dy - 2) * L_curried(3)
                + dy * (dy + 1) * (dy - 1) * L_curried(4) / 3) / 2
    return BLANK_VALUE


# The mapping is indirect to prevent blank regions:
def indirect_mapping(f_prime, T_prime, method):
    w_prime, h_prime = f_prime.shape
    for x_prime in range(w_prime):
        for y_prime in range(h_prime):
            pos_prime = np.array((x_prime, y_prime, 1))  # homogeneous
            # Indirect of the position:
            x, y, _ = T_prime @ pos_prime  # equivalent to: P = T'P'
            # Gets the output value using the given interpolation method:
            f_prime[x_prime][y_prime] = method(x, y)


interpolations = [closest, bilinear, bicubic, lagrange]
if __name__ == '__main__':
    start = time.time()
    args = read_args()

    # Prepare the data:
    in_img = img_read(args.i)
    w, h = in_img.shape
    method = partial(interpolations[args.m], in_img, w, h)

    w_prime, h_prime = w, h
    if args.a is not None:
        T_prime = rot_mat(args.a, w / 2, h / 2)  # rotate around the center
    elif args.d:
        T_prime = scl_mat(args.d[0] / w, args.d[1] / h)
        w_prime, h_prime = args.d
    else:
        T_prime = scl_mat(args.e, args.e)
        w_prime, h_prime = int(args.e * w), int(args.e * h)

    # Transforms the image:
    result = np.empty((w_prime, h_prime), dtype=np.uint8)
    indirect_mapping(result, T_prime, method)
    save_img(result, args.o)

    end = time.time()
    print('Time elapsed: {:.2f}s'.format(end - start))
