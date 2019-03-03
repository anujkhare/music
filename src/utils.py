
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import montage

from utils.constants import Strings


def visualize_one(images, target_label, pred_labels):
    pad_idxs = np.where(target_label == -100)
    target_label[pad_idxs] = 0
    pred_labels[pad_idxs] = 0

    images = images.astype(np.int)

    ndim = images.shape[0]
    if ndim > 1:
        images = montage(images, multichannel=True)
        target_label = montage(target_label)
        pred_labels = montage(pred_labels)
    else:
        images = images[0]
        target_label = target_label[0]
        pred_labels = pred_labels[0]

    N = 3
    plt.subplot(1, N, 1)
    plt.imshow(images)
    plt.title('Input images')

    plt.subplot(1, N, 2)
    plt.imshow(target_label)
    plt.title('Targets')

    plt.subplot(1, N, 3)
    plt.imshow(pred_labels)
    plt.title('Predictions')


def visualize_predictions(images, target_arrs, pred_arrs, titles, figsize=None):
    figsize = figsize or (20, 10)
    N = len(target_arrs)
    if len(pred_arrs) != N and len(titles) != N:
        raise ValueError

    images = images.transpose(0, 2, 3, 1)

    for target, pred, title in zip(target_arrs, pred_arrs, titles):
        plt.figure(figsize=figsize)  # FIXME: Should pass an `ax` to the next function...
        print(title)
        if title == Strings.image_label:
            print('Target: {}, Pred: {}'.format(target, pred))
            continue
        visualize_one(images, target_label=target, pred_labels=pred)
        plt.show()