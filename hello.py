import numpy as np
from PIL import Image


def normalize(value, min, max):
    normalized = (value - min) / (max - min)  # Normalize to [0, 1]
    return int(normalized * 255.0).to_bytes(1)


def main():
    print("Hello from gpr-viewer!")
    data_file = np.loadtxt("data/FILE____010.ASC")
    assert (
        data_file.shape[0] % 1024 == 0
    ), "Data must be able to be seperated into chunks of 1024 evenly"

    bscan = data_file.reshape((1024, -1, 4))
    print(bscan[:, :, 3][0][:50])

    min = np.min(bscan[:, :, 3].flatten())
    max = np.max(bscan[:, :, 3])
    print(f"min: {min}")
    print(f"max: {max}")
    vec_normalize = np.vectorize(normalize)

    mapped_bscan = vec_normalize(bscan[:, :, 3], min, max)
    print(mapped_bscan.flatten())
    print(mapped_bscan.dtype)
    print(mapped_bscan.shape)

    # image = Image.fromarray(mapped_bscan, "L")
    image = Image.frombytes("L", mapped_bscan.shape, mapped_bscan.flatten())
    image.save("out2.png")


if __name__ == "__main__":
    main()
