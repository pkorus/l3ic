#!/usr/bin/python3
import os
import imageio
import argparse
import numpy as np

from datetime import datetime
from skimage.measure import compare_ssim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.dcn import DCN
from helpers import afi


bitmap_formats = {'.png', '.jpg', '.bmp', '.jpeg', '.jp2'}

parser = argparse.ArgumentParser(description='Show results from NIP & FAN optimization')
parser.add_argument('-v', '--verbose', action='store_true', help='run additional tests (AE benchmark)')
parser.add_argument('-d', '--dir', default='./samples/clic512', help='input directory (bitmaps only)')
parser.add_argument('-m', '--model', dest='model', action='store', default='32c',
                    help='DCN model - corresponds to quality: 16c, 32c, 64c')

args = parser.parse_args()

# List available images
files = [x for x in os.listdir(args.dir) if os.path.splitext(x)[-1].lower() in bitmap_formats]
print('Found {} test images in {}'.format(len(files), args.dir))

# Create an instance of the NN model
dcn = DCN(args.model)

stats = {
    'compress': [], 
    'decompress': [], 
    'encode': [], 
    'decode': [], 
    'ssim': [], 
    'bpp': [], 
    'bytes': [],
    'shape': []
}

for filename in files:

    # Load the image
    image = imageio.imread(os.path.join(args.dir, filename), pilmode='RGB').astype(np.float32) / 255.0
    stats['shape'].append(image.shape[:2])

    # Process
    t1 = datetime.now()
    compressed_stream = afi.compress(dcn, np.expand_dims(image, axis=0))
    t2 = datetime.now()
    decompressed_imag = afi.decompress(dcn, compressed_stream).squeeze()
    t3 = datetime.now()
    ssim = compare_ssim(image, decompressed_imag, multichannel=True, data_range=1.0)

    if args.verbose:
        t4 = datetime.now()
        latent = dcn.compress(np.expand_dims(image, axis=0))
        t5 = datetime.now()
        dcn.decompress(latent)
        t6 = datetime.now()

    stats['compress'].append((t2 - t1).total_seconds())
    stats['decompress'].append((t3 - t2).total_seconds())

    if args.verbose:
        stats['encode'].append((t5 - t4).total_seconds())
        stats['decode'].append((t6 - t5).total_seconds())

    stats['ssim'].append(ssim)
    stats['bytes'].append(len(compressed_stream))
    stats['bpp'].append(8 * len(compressed_stream) / image.shape[0] / image.shape[1])

    assert ssim > 0.5, 'There seems to be an issue with the model - poor SSIM detected ({})'.format(ssim)

print('\n# DCN model ({}):'.format(args.model))
for id, filename in enumerate(files):
    print('{:>30s} {} -> ssim {:.3f} @ {:.2f} bpp'.format(filename, stats['shape'][id], stats['ssim'][id], stats['bpp'][id]))

print('\n# Average processing time:')
print('  full compression   : {:.4f} s'.format(np.mean(stats['compress'])))
print('  full decompression : {:.4f} s'.format(np.mean(stats['decompress'])))
if args.verbose:
    print('  deep encoding      : {:.4f} s'.format(np.mean(stats['encode'])))
    print('  deep decoding      : {:.4f} s'.format(np.mean(stats['decode'])))
