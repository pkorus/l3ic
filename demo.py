import argparse
import numpy as np
import imageio
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.measure import compare_ssim

from models.dcn import DCN
from format import afi

def quickshow(ax, image, title):
    ax.imshow(image.squeeze())
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


parser = argparse.ArgumentParser(description='Show results from NIP & FAN optimization')
parser.add_argument('-i', '--image', default='./samples/md575e5a225f.png')
parser.add_argument('-m', '--model', dest='model', action='store', default='8k',
                    help='DCN model - corresponds to quality (4k, 8k, 16k)')
parser.add_argument('-s', '--stats', dest='stats', action='store_true', default=False,
                    help='Show detailed stats')

args = parser.parse_args()

image = imageio.imread(args.image)
image = image.astype(np.float32) / 255
image = np.expand_dims(image, axis=0)

dcn = DCN(args.model)

if args.stats:
    # Some benchmarking
    t1 = datetime.now()
    compressed = dcn.process(image)
    t2 = datetime.now()
    latent = dcn.compress(image)
    t3 = datetime.now()
    decompressed = dcn.decompress(latent)
    t4 = datetime.now()
    afi.compress(dcn, image)
    t5 = datetime.now()
    fully_decoded, stats = afi.compress_n_stats(dcn, image)
    t6 = datetime.now()

    print('Latent space: ', latent.shape)
    print('DCN Simulation (Enc+Dec) time:', (t2 - t1).total_seconds(), 's')
    print('DCN Encoding time: ', (t3 - t2).total_seconds(), 's')
    print('DCN Decoding time: ', (t4 - t3).total_seconds(), 's')
    print('Full AFI encoding: ', (t5 - t4).total_seconds(), 's')
    print('Stats:')
    for k, v in stats.items():
        print('  {}: {:.3f}'.format(k, v))

    fig, axes = plt.subplots(1, 3)
    quickshow(axes[0], image, 'Input')
    quickshow(axes[1], compressed, 'Simulated (DCN) ssim={:.2f}'.format(stats['ssim']))
    quickshow(axes[2], fully_decoded, 'Full encoding / decoding')
    fig.tight_layout()
    plt.show()

else:
    t1 = datetime.now()
    compressed, image_bytes = afi.simulate_compression(dcn, image)
    t2 = datetime.now()
    ssim = compare_ssim(image.squeeze(), compressed.squeeze(), multichannel=True, data_range=1.0)
    
    print('Full compression + decompression time:', (t2 - t1).total_seconds(), 's')
    print('Bitstream: {:,} bytes ({:.3f} bpp)'.format(image_bytes, 8 * image_bytes / image.shape[1] / image.shape[2]))
    print('SSIM: {:.3f}'.format(ssim))
    fig, axes = plt.subplots(1, 2)
    quickshow(axes[0], image, 'Input')
    quickshow(axes[1], compressed, 'Simulated (DCN) ssim={:.2f}'.format(ssim))
    fig.tight_layout()
    plt.show()
