import pickle
from collections import Counter

import pyfse
import dahuffman
from utils import short_string, entropy, symbol_probabilities


with open('tests/string.txt') as f:
    input = f.read().encode('ascii')

n = len(input)

print('Input [{:,} symbols]: {}'.format(len(input), short_string(input)))

# Gather stats
input_stats = dict(Counter(input))
input_prob = symbol_probabilities(input)
input_entropy = entropy(input_prob.values())

print('Frequency stats:')
print('  Symbol probabilities: {}'.format({k: round(v, 2) for k, v in input_prob.items()}))
print('  Entropy: {:.2f}'.format(input_entropy))

# Huffman codec
codec = dahuffman.HuffmanCodec.from_data(input_stats)
coded_huff = codec.encode(input)

print('Huffman code table: {}'.format(codec.get_code_table()))

print('Theoretical length: {:,.1f} bytes'.format(len(input) * input_entropy / 8))
print('Huffman coded: {:,} bytes ({:,} including pickled p. tables)'.format(len(coded_huff),
    len(pickle.dumps(codec.get_code_table())) + len(coded_huff)
))

# The ANS codec
coded_fse = pyfse.easy_compress(input)
decoded_fse = pyfse.easy_decompress(coded_fse)

print('FSE decoding success: {}'.format(input == decoded_fse))
print('FSE coded: {:,} bytes (includes codec p. tables)'.format(len(coded_fse)))
print('FSE decoded [{:,}]: {}'.format(len(decoded_fse), short_string(decoded_fse)))
