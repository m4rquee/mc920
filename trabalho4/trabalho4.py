#!/usr/bin/env python
import time
import argparse

import numpy as np
from skimage import io

NUM_BYTES = 4  # number of bytes in the size header


def read_args():
    parser = argparse.ArgumentParser(description='Image steganography.')
    parser.add_argument('-i', required=True,
                        help='The name of the input image.')
    parser.add_argument('-f', required=False,
                        help='The name of the input file.')
    parser.add_argument('-o', required=True,
                        help='The name of the output file.')
    parser.add_argument('-p', required=False, default=0, type=int,
                        help='The bit plane to use.')
    parser.add_argument('-e', required=False, choices=range(2), default=0,
                        type=int, help='If should encode (0) or decode (1).')
    return parser.parse_args()


def img_read(int_name):
    return io.imread(int_name).astype(np.uint8)


def save_img(out_img, out_name):
    io.imsave(out_name, out_img.astype(np.uint8), check_contrast=False)


def read_file(name):
    with open(name, "rb") as file:
        content = list(file.read())
        size = len(content)
    return content, size


def write_file(content, name):
    with open(name, "wb") as file:
        arr = bytearray(content)
        file.write(arr)


def encode(img, content, size, plane):
    # Checks if the file fits in the image:
    bit_file_size = 8 * (size + NUM_BYTES)
    if img.size < bit_file_size:
        raise Exception("File is too big!")

    # Adds the size header:
    size_bytes = list(size.to_bytes(NUM_BYTES, 'big'))
    content = np.array(size_bytes + content, dtype=np.uint8)

    bit_file = np.unpackbits(content)  # unpacks each byte to a binary form

    # Creates the mask/values by shifting to the right plane:
    bit_values = bit_file << plane
    mask = np.full(bit_file_size, 1 << plane)

    # Pads the mask/values to be able to do a simple apply later:
    left_over = img.size - bit_file_size  # padding length to reach the original
    bit_values = np.pad(bit_values, (0, left_over), constant_values=0)
    mask = np.pad(mask, (0, left_over), constant_values=0)

    # Reshapes to match the original image:
    bit_values = bit_values.reshape(img.shape)
    mask = mask.reshape(img.shape)

    # Applies the mask:
    result = img & (255 - mask)  # erases the bits of the plane inside the mask
    result |= bit_values  # writes the new values
    return result


def decode(img, plane):
    start = 7 - plane  # the first file bit in the flat unpacked array
    bit_file = np.unpackbits(img)[start::8]  # gets only the right plane
    file = np.packbits(bit_file)  # converts the bit array to a byte array
    # The size is the first NUM_BYTES bytes:
    size = int.from_bytes(file[:NUM_BYTES], byteorder='big', signed=False)
    return file[NUM_BYTES:NUM_BYTES + size]  # crops the file out


if __name__ == '__main__':
    start = time.time()
    args = read_args()
    in_img = img_read(args.i)

    if args.e == 0:
        content, size = read_file(args.f)
        result = encode(in_img, content, size, args.p)
        save_img(result, args.o)
    else:
        result = decode(in_img, args.p)
        write_file(result, args.o)

    end = time.time()
    print('Time elapsed: {:.2f}ms'.format(1E3 * (end - start)))
