#!/usr/bin/env python
import time
import argparse

import numpy as np
from skimage import io


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
    parser.add_argument('-d', required=False, type=int,
                        help='Dimension of the output image in pixels.')
    parser.add_argument('-m', required=False, default=0, choices=range(4),
                        help='Interpolation method to use.')
    ret = parser.parse_args()

    if all(v is None for v in [ret.a, ret.e, ret.d]):
        parser.error('No action requested! Add -a, -e or -d.')
    return ret


def img_read(in_name):
    return io.imread(in_name, as_gray=True).astype(np.uint8)


def save_img(out_img, out_name):
    io.imsave(out_name, out_img.astype(np.uint8), check_contrast=False)


if __name__ == '__main__':
    start = time.time()
    args = read_args()
    in_img = img_read(args.i)

    end = time.time()
    print('Time elapsed: {:.2f}s'.format(end - start))
