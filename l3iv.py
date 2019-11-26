#!/usr/bin/python3
import os
import sys
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.dcn import DCN
from helpers import codec

file_ext = '.l3i'
bitmap_formats = {'.png', '.jpg', '.bmp', '.jpeg', '.jp2'}

def quickshow(ax, image, title):
    ax.imshow(image.squeeze())
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

parser = argparse.ArgumentParser(description='Viewer for Lightweight Learned Lossy Image Codec (l3ic)')
parser.add_argument('input', help='Coded image *.{})'.format(file_ext))
parser.add_argument('-m', '--model', dest='model', action='store', default='32c',
                    help='DCN model - corresponds to quality: 16c, 32c, 64c')

args = parser.parse_args()

if args.input is None:
    parser.print_usage()
    sys.exit(1)

dcn = DCN(args.model)

if os.path.splitext(args.input)[-1].lower() == file_ext:

    with open(args.input, 'rb') as f:
        coded_stream = f.read()
        image = codec.decompress(dcn, coded_stream)

    fig = plt.figure()
    quickshow(fig.gca(), image, args.input)
    fig.tight_layout()
    plt.show()

else:
    print('The file extension is not supported!')
    sys.exit(1)