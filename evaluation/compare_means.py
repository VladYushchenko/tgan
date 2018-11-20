import argparse
import numpy as np

def is_identical(args):
    """Compares full and cropped mean files"""

    cropped_mean = np.load(args.mean_cropped).astype('f')
    cropped_mean = np.expand_dims(cropped_mean, 0)

    # NOTE: commenting this line make mean files be unequal
    cropped_mean = cropped_mean.transpose((4, 0, 1, 2, 3))
    cropped_mean = cropped_mean.reshape((3, 1, 16, 112, 112))

    full_mean = np.load(args.mean_full).astype('f')
    full_mean = full_mean.reshape((3, 1, 16, 128, 171))[:, :, :, :, 21:21 + 128]
    full_mean = full_mean[:, :, :, 8:8 + 112, 8:8 + 112]

    return np.all(cropped_mean == full_mean) and (cropped_mean.shape == full_mean.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--mean_full')
    parser.add_argument('--mean_cropped')

    args = parser.parse_args()

    print('Is Two mean files identical: {}'.format(is_identical(args)))