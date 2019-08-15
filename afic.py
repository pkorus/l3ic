#!/usr/bin/python3
import os
import sys
import imageio
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.dcn import DCN
from helpers import afi

afi_ext = '.afi'
bitmap_formats = {'.png', '.jpg', '.bmp', '.jpeg', '.jp2'}

parser = argparse.ArgumentParser(description='AFI codec')
parser.add_argument('-i', '--input', dest='input', 
                    help='Input file (bitmap or AFI stream)')
parser.add_argument('-m', '--model', dest='model', action='store', default='8k',
                    help='DCN model - corresponds to quality: 4k, 8k, 16k')
parser.add_argument('-o', '--output', dest='output', action='store', 
                    help='Output file (bitmap or AFI stream)')

args = parser.parse_args()

if args.input is None:
    parser.print_usage()
    sys.exit(1)

dcn = DCN(args.model)

if os.path.splitext(args.input)[-1].lower() == afi_ext:

    with open(args.input, 'rb') as f:
        coded_stream = f.read()
        image = afi.decompress(dcn, coded_stream)

    if args.output is None:
        args.output = args.input.replace(afi_ext, '.png')

    imageio.imwrite(args.output, (255 * image.squeeze()).astype(np.uint8))

if os.path.splitext(args.input)[-1].lower() in bitmap_formats:

    image = imageio.imread(args.input).astype(np.float32) / 255
    image = np.expand_dims(image, axis=0)

    coded_stream = afi.compress(dcn, image)

    if args.output is None:
        args.output = args.input.replace(os.path.splitext(args.input)[-1], '.afi')

    with open(args.output, 'wb') as f:
        f.write(coded_stream)

