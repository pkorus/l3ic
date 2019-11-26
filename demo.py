#!/usr/bin/python3
import os
import imageio
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
from datetime import datetime
from skimage.measure import compare_ssim

from models.dcn import DCN
from helpers import codec


def quickshow(ax, image, title):
    ax.imshow(image.squeeze())
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


parser = argparse.ArgumentParser(description='Show results from NIP & FAN optimization')
parser.add_argument('-i', '--image', default='./samples/kodim05.png')
parser.add_argument('-m', '--model', dest='model', action='store', default='32c',
                    help='DCN model - corresponds to quality (16c, 32c, 64c)')
parser.add_argument('-s', '--stats', dest='stats', action='store_true', default=False,
                    help='Show detailed stats')

args = parser.parse_args()

image = imageio.imread(args.image).astype(np.float32) / 255
image = np.expand_dims(image, axis=0)

dcn = DCN(args.model)

t1 = datetime.now()
compressed, image_bytes = codec.simulate_compression(dcn, image)
t2 = datetime.now()
ssim = compare_ssim(image.squeeze(), compressed.squeeze(), multichannel=True, data_range=1.0)
bpp = 8 * image_bytes / image.shape[1] / image.shape[2]

print('Full compression + decompression time :', (t2 - t1).total_seconds(), 's')
print('Bitstream                             : {:,} bytes ({:.3f} bpp)'.format(image_bytes, bpp))
print('SSIM                                  : {:.3f}'.format(ssim))
fig, axes = plt.subplots(1, 2)
quickshow(axes[0], image, 'Input')
quickshow(axes[1], compressed, 'Compressed ({}) ssim={:.2f} @ {:.2f} bpp'.format(args.model, ssim, bpp))
fig.tight_layout()
plt.show()
