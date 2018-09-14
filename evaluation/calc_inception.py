#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np

import chainer
import chainer.cuda
import cv2 as cv
from c3d_ft import C3DVersion1
from chainer import Variable
from chainer import cuda
from tqdm import tqdm

sys.path.insert(0, '/home/vlad/PycharmProjects/MasterThesis/src')
from loaders import Loader


def calc_inception(ys):
    N, C = ys.shape
    p_all = np.mean(ys, axis=0, keepdims=True)
    kl = np.sum(ys * np.log(ys + 1e-7) - ys * np.log(p_all + 1e-7)) / N
    return np.exp(kl)

def get_inception(c3dmodel, args):
    """Launch chainer and extract inception score"""
    xp = chainer.cuda.cupy
    mean = np.load(args.mean)
    mean = mean.reshape((3, 1, 16, args.image_size, args.image_size))
    mean = np.mean(mean, axis=2, keepdims=True)
    loader = Loader(args)()

    ys = []
    for _, (x, _) in tqdm(enumerate(loader), total=len(loader)):
        x = x.data.cpu().numpy()
        # x = x.transpose(1, 0, 2, 3, 4)
        # x = (x + 1) / 2 * 255
        # # Needed to equal images loaded from disk vs directly generated images
        # # While saving from float32 to uint8 [0 .. 255], using this normalization
        # # x = np.around(x).astype('uint8')
        # x = x.astype('uint8')

        # TODO: use for direct comparison with TGAN paper (use INTER_CUBIC)
        n, c, f, h, w = x.shape
        x = x.transpose(0, 2, 3, 4, 1).reshape(n * f, h, w, c)
        x = x * 128 + 128
        x_ = np.zeros((n * f, args.image_size, args.image_size, 3))
        for t in range(n * f):
            x_[t] = np.asarray(
                cv.resize(x[t], (args.image_size, args.image_size), interpolation=args.interpolation))
        x = x_.transpose(3, 0, 1, 2).reshape(3, n, f, args.image_size, args.image_size)

        # mean file is BGR-order while model outputs RGB-order
        x = x[::-1] - mean
        x = x.transpose(1, 0, 2, 3, 4)

        with chainer.using_config('train', False) and chainer.no_backprop_mode():
            # C3D takes an image with BGR order
            y = c3dmodel(Variable(xp.asarray(x, dtype=np.float32)), layers=['prob'])['prob'].data.get()
            ys.append(y)

    ys = np.asarray(ys).reshape((-1, 101))
    return ys


def calculate_inception_score(args):
    np.random.seed(args.seed)

    args.interpolation = getattr(cv, args.interpolation)

    cuda.get_device(args.device).use()
    chainer.cuda.cupy.random.seed(args.seed)

    c3dmodel = C3DVersion1()
    c3dmodel.to_gpu()

    ys = get_inception(c3dmodel, args)

    score = calc_inception(ys)
    print(score)

    # with open('{}/inception_iter-{}_{}.txt'.format(args.result_dir, args.iter, inter_method), 'w') as fp:
    #     print(args.result_dir, args.iter, args.calc_iter, args.mean, score, file=fp)
    #     print(args.result_dir, args.iter, 'score:{}'.format(score))
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tgan inception score')
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--result_dir', default='./result')
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--iter', type=int, default=100000)
    parser.add_argument('--calc_iter', type=int, default=10000)
    parser.add_argument('--mean', type=str, default='/home/vlad/Downloads/crop_mean.npy')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--video_length', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ext', default='jpg')
    parser.add_argument('--interpolation', type=str, default='INTER_CUBIC')
    args = parser.parse_args()

    calculate_inception_score(args)
