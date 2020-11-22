#!/usr/bin/env python
import time
import argparse

import numpy as np
from skimage import io

# Error distribution masks:
mask_a = np.array([
    [0, 0, 7],
    [3, 5, 1]
]) / 16
mask_b = np.array([
    [0, 0, 0, 0, 0, 32, 0],
    [12, 0, 26, 0, 30, 0, 16],
    [0, 12, 0, 26, 0, 12, 0],
    [5, 0, 12, 0, 12, 0, 5]
]) / 200
mask_c = np.array([
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2]
]) / 32
mask_d = np.array([
    [0, 0, 0, 5, 3],
    [2, 4, 5, 4, 2],
    [0, 2, 3, 2, 0]
]) / 32
mask_e = np.array([
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2],
    [1, 2, 4, 2, 1]
]) / 42
mask_f = np.array([
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1]
]) / 48
masks = [mask_a, mask_b, mask_c, mask_d, mask_e, mask_f]


def read_args():
    parser = argparse.ArgumentParser(description='Image halftoning.')
    parser.add_argument('-i', required=True,
                        help='The name of the input image.')
    parser.add_argument('-o', required=False, default='./out.png',
                        help='The name of the output image.')
    parser.add_argument('-num', required=True, choices=range(1, 7), type=int,
                        help='The error mask number.')
    parser.add_argument('-zig', required=False, default=False, type=bool,
                        help='Whether to zigzag while sweeping the image.')
    parser.add_argument('-k', required=False, default=2, type=int,
                        help='How many levels to use in each color channel.')
    return parser.parse_args()


def img_read(args):
    return io.imread(args.i).astype(float)


def halftone(img, mask, zig, k):
    il, ic, _ = img.shape  # lines, columns
    ml, mc = mask.shape  # lines, columns
    border = (mc - 1) // 2  # number of pixels outside the image

    # Extends the image with zeros:
    resized = np.zeros((il + ml - 1, ic + mc - 1, 3), dtype=float)
    resized[0:il, border:ic + border] = img

    # Creates the ranges to sweep the image:
    norm_ran = range(border, ic + border)  # from left to right
    if zig:
        flip_ran = range(ic + border - 1, border - 1, -1)  # from right to left

    # Replicate the mask to use in each channel:
    norm_mmask = np.stack((mask, mask, mask), axis=2)
    if zig:
        flip_mask = np.flip(mask, 1)  # mask used in the odd iterations
        flip_mmask = np.stack((flip_mask, flip_mask, flip_mask), axis=2)

    ran = norm_ran
    mmask = norm_mmask  # multi-mask (to each channel)
    f0, f1 = k / 255, 255 / k  # used to discretize the pixels
    for i in range(il):
        for j in ran:
            # Discretizes the pixel into k levels:
            new_pixel = np.round(f1 * np.round(f0 * resized[i][j]))
            error = resized[i][j] - new_pixel
            resized[i][j] = new_pixel

            carry = error * mmask  # distribute the error using the mask
            # Propagates the error:
            resized[i:i + ml, (j - border):(j + border + 1)] += carry

        # Flips the range/mask while zigzagging:
        if zig:
            ran = flip_ran if i % 2 == 0 else norm_ran
            mmask = flip_mmask if i % 2 == 0 else norm_mmask

    return resized[0:il, border:ic + border]  # crops the axillary borders


def save_img(out_img, out_name):
    out_img = out_img.round().clip(0, 255).astype(np.uint8)
    io.imsave(out_name, out_img, check_contrast=False)


if __name__ == '__main__':
    start = time.time()
    args = read_args()
    in_img = img_read(args)
    # print(masks[args.num - 1])
    result = halftone(in_img, masks[args.num - 1], args.zig, args.k - 1)
    # print(np.unique(result))
    save_img(result, args.o)
    end = time.time()
    print("Time elapsed: {:.4f}s".format(end - start))
