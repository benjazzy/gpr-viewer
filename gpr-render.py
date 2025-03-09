import argparse
import math
import os
from fileinput import filename

import h5py
import numpy as np
from PIL import Image


def normalize(value, min, max):
    normalized = (value - min) / (max - min)  # Normalize to [0, 1]
    # print(int(normalized * 255.0))
    return int(normalized * 255.0).to_bytes(1)


def parse_asc(file_name):
    data_file = np.loadtxt(file_name)
    assert (
        data_file.shape[0] % 1024 == 0
    ), "Data must be able to be seperated into chunks of 1024 evenly"

    print("Transforming")
    bscan = data_file.reshape((-1, 1024, 4))
    bscan = bscan[:, :, 3].reshape((-1, 1024))

    min = np.min(bscan.flatten())
    max = np.max(bscan)
    vec_normalize = np.vectorize(normalize)
    normalized = vec_normalize(bscan, min, max)
    rotated = np.rot90(normalized, 3)

    return rotated


def parse_hdf5(file_name):
    with h5py.File(file_name) as data_file:
        bscan = data_file["rxs"]["rx1"]["Ex"]

        min = np.min(bscan)
        max = np.max(bscan)
        vec_normalize = np.vectorize(normalize)
        return vec_normalize(bscan[:, :], min, max)


def parse(file_name, aspect_ratio):
    if not os.path.isfile(file_name):
        raise FileNotFoundError(file_name)

    bscan = None

    extension = os.path.splitext(file_name)[1]
    if extension == ".ASC":
        print("Reading ASC")
        bscan = parse_asc(file_name)
    elif extension == ".out":
        print("Reading hdf5")
        bscan = parse_hdf5(file_name)
    else:
        print(f"Unkown extension: {extension}")
        raise ValueError

    if aspect_ratio:
        data_height, data_width = bscan.shape
        data_ratio = data_width / data_height
        print(data_ratio)
        ratio_multiplier = round(aspect_ratio / data_ratio)
        print(ratio_multiplier)
        if ratio_multiplier >= 2:
            bscan = bscan.repeat(ratio_multiplier, axis=1)

    return bscan


def main():
    parser = argparse.ArgumentParser(prog="gpr-render")
    parser.add_argument("filename")
    parser.add_argument("-a", "--aspect-ratio", required=False)
    args = parser.parse_args()
    name = os.path.splitext(args.filename)[0]
    aspect_ratio = None
    if args.aspect_ratio:
        width, height = args.aspect_ratio.split(":")
        aspect_ratio = float(width) / float(height)

    bscan = parse(args.filename, aspect_ratio)
    print(f"Writing output to {name}.png")
    image = Image.fromarray(bscan, "L")
    image.save(f"{name}.png")


if __name__ == "__main__":
    main()
