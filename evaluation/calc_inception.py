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

sys.path.insert(0, '')
from loaders import Loader


def calc_inception(ys):
    N, C = ys.shape
    p_all = np.mean(ys, axis=0, keepdims=True)
    kl = np.sum(ys * np.log(ys + 1e-7) - ys * np.log(p_all + 1e-7)) / N
    # expreimental
    # mean_scores = ys * np.log(ys + 1e-7) - ys * np.log(p_all + 1e-7)
    # mean_scores = mean_scores.transpose().tolist()
    # mean_scores = [sorted(item, reverse=True)[:max_vid] for item in mean_scores]
    # mean_scores = np.asarray(mean_scores).transpose()
    # # mean_top_scores = np.sort(mean_scores)[:, ::-1][:, :max_vid]
    # kl = np.sum(mean_scores) / N
    return np.exp(kl)

def get_inception(c3dmodel, args):
    """Launch chainer and extract inception score"""
    xp = chainer.cuda.cupy
    # mean = np.zeros((3, 1, 16, 128, 128))
    # loaded_mean = np.load(args.mean).astype('f')
    # print(loaded_mean[0][0][0])
    # loaded_mean = np.expand_dims(loaded_mean, 0)
    # # loaded_mean = loaded_mean.transpose((4, 0, 1, 2, 3))
    # loaded_mean = loaded_mean.reshape((3, 1, 16, 112, 112))
    # print(loaded_mean[:, 0, 0, 0, 0])
    # mean[:, :, :, 8:120, 8:120] = loaded_mean

    loaded_mean = np.load(args.mean).astype('f')
    loaded_mean = loaded_mean.reshape((3, 1, 16, 128, 171))[:, :, :, :, 21:21 + 128]
    print(loaded_mean[:, 0, 0, 0, 0])
    mean = loaded_mean

    loader = Loader(args)()

    ys = []
    for _, (x, _) in tqdm(enumerate(loader), total=len(loader)):
        x = x.data.cpu().numpy()
        n, c, f, h, w = x.shape
        x = x.transpose(0, 2, 3, 4, 1).reshape(n * f, h, w, c)
        x = x * 128 + 128
        x_ = np.zeros((n * f, 128, 128, 3))
        for t in range(n * f):
            x_[t] = np.asarray(
                cv.resize(x[t], (128, 128), interpolation=args.interpolation))
        x = x_.transpose(3, 0, 1, 2).reshape(3, n, f, 128, 128)
        x = x[::-1] - mean  # mean file is BGR-order while model outputs RGB-order
        x = x[:, :, :, 8:8 + 112, 8:8 + 112].astype('f')
        x = x.transpose(1, 0, 2, 3, 4)
        with chainer.using_config('train', False) and \
             chainer.no_backprop_mode():
            # C3D takes an image with BGR order
            y = c3dmodel(Variable(xp.asarray(x)),
                         layers=['prob'])['prob'].data.get()
            ys.append(y)
    ys = np.asarray(ys).reshape((-1, 101))
    return ys


def calculate_inception_score(args):
    if args.seed >= 0:
        np.random.seed(args.seed)
        chainer.cuda.cupy.random.seed(args.seed)

    cuda.get_device(args.device).use()

    c3dmodel = C3DVersion1()
    c3dmodel.to_gpu()

    ys = get_inception(c3dmodel, args)

    score = calc_inception(ys)
    print(score)

    with open('{}/inception_iter-_{}.txt'.format(args.result_dir, args.interpolation), 'w') as fp:
        print(args.result_dir,  args.calc_iter, args.mean, score, file=fp)
        print(args.result_dir, 'score:{}'.format(score))
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tgan inception score')
    parser.add_argument('--location', type=str)
    parser.add_argument('--mode')
    parser.add_argument('--result_dir', default='./result')
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--calc_iter', type=int, default=10000)
    parser.add_argument('--mean', type=str, default='./converted_ucf_mean.npy')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--video_length', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ext', default='jpg')
    parser.add_argument('--interpolation', type=str, default='INTER_CUBIC')
    args = parser.parse_args()
    args.interpolation = getattr(cv, args.interpolation)

    scores = []
    total_test = 1
    #for _ in tqdm(range(total_test), total=total_test):
    for _ in range(total_test):
        score = calculate_inception_score(args)
        scores.append(score)

    scores = np.asarray(scores)
    print(np.max(scores), np.mean(scores), np.std(scores))
