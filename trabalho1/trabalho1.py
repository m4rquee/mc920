#!/usr/bin/env python
import time
import argparse

import numpy as np
from skimage import io
from scipy import ndimage
from skimage.util import img_as_ubyte

# Default kernels:
h1 = np.array([
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0]
])
h2 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]) / 256
h3 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
h4 = h3.T
h5 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])
h6 = np.ones((3, 3)) / 9
h7 = np.array([
    [-1, -1, 2],
    [-1, 2, -1],
    [2, -1, -1]
])
h8 = np.array([
    [2, -1, -1],
    [-1, 2, -1],
    [-1, -1, 2]
])
h9 = np.identity(9) / 9
h10 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 2, 2, 2, -1],
    [-1, 2, 8, 2, -1],
    [-1, 2, 2, 2, -1],
    [-1, -1, -1, -1, -1]
]) / 8
h11 = np.array([
    [-1, -1, 0],
    [-1, 0, 1],
    [0, 1, 1]
])
h12 = np.ones((9, 9)) / 81 # aditional
h = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12]


def read_args():
    parser = argparse.ArgumentParser(description='Convolution filtering.')
    parser.add_argument('-i', required=True,
                        help='The name of the input image.')
    parser.add_argument('-o', required=False, default='./out.png',
                        help='The name of the output image.')
    parser.add_argument('-num', required=False, choices=range(1, 13), type=int,
                        help='The kernel number.')
    parser.add_argument('-m', required=False, default='nearest', choices=[
        'reflect', 'constant', 'nearest', 'mirror', 'wrap'
    ], help='The mode parameter determines how the array borders are handled.')
    parser.add_argument('-s', required=False, type=str,
                        help='The custom kernel shape.')
    parser.add_argument('-k', required=False, type=str,
                        help='The custom kernel values')
    parser.add_argument('-bit', required=False, default=-1, choices=range(0, 8),
                        type=int, help='The selected bit plane.')
    parser.add_argument('-r', required=False, default=False, type=bool,
                        help='If the output range should be resized.')
    parser.add_argument('-h3h4', required=False, default=False, type=bool,
                        help='If the h3 and h4 combination should be used.')
    ret = parser.parse_args()

    if not (ret.num or ret.k or ret.h3h4):
        parser.error('No action requested! Add -num, -k or -h3h4.')

    if ret.s: ret.s = tuple(map(int, ret.s.split(',')))
    if ret.k: ret.k = np.fromiter(ret.k.split(','), float).reshape(ret.s)
    return ret


def bit_plane(img, bit=7):
    return (img & (1 << bit)) >> bit


def img_read(args):
    in_img = img_as_ubyte(io.imread(args.i, as_gray=True))
    if args.bit != -1: in_img = bit_plane(in_img, args.bit)
    return in_img.astype(float)


def resize_range(img):
    maxi = max(np.max(img), 1)
    return 255.0 / maxi * img


def process_img(in_img, args):
    weights = h[args.num - 1] if args.num else args.k
    return ndimage.convolve(in_img, weights, mode=args.m)


def save_img(out_img, out_name):
    out_img = out_img.round().clip(0, 255).astype(np.uint8)
    io.imsave(out_name, out_img, check_contrast=False)


def h3h4(in_img, args):
    h32 = ndimage.convolve(in_img, h3, mode=args.m) ** 2
    h42 = ndimage.convolve(in_img, h4, mode=args.m) ** 2
    return np.sqrt(h32 + h42)


if __name__ == '__main__':
    start = time.time()
    args = read_args()
    in_img = img_read(args)
    result = h3h4(in_img, args) if args.h3h4 else process_img(in_img, args)
    if args.r: result = resize_range(result)
    save_img(result, args.o)
    end = time.time()
    print("Time elapsed: {:.4f}s".format(end - start))
