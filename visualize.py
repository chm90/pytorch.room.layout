import sys

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
import numpy as np
from imageio import imread
from os import path, listdir
import scipy.io as spio

for res_fn in listdir('features/'):
    feature = spio.loadmat(path.join('features', res_fn), squeeze_me=False)['feature']
    feature = np.transpose(feature, [2, 3, 1, 0])
    feature = feature[:, :, 1:3]
    feature = np.max(feature, axis=2)
    im = imread(path.join('datasets/lsun/images/', res_fn[:-3] + 'png'))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im)
    ax2.imshow(np.exp(feature.squeeze()))
    plt.show()
