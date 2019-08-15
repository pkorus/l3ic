import unittest
import pyfse
import utils


class EasyTest(unittest.TestCase):

    def setUp(self) -> None:
        self.inputs = {}

        with open('tests/string.txt') as f:
            self.inputs['ascii'] = f.read().encode('ascii')

        with open('tests/binary.dat') as f:
            self.inputs['binary'] = bytes([int(x) for x in f.read()])

        with open('tests/numbers.dat') as f:
            self.inputs['numbers'] = bytes([127 + int(x) for x in f.read().split(', ')])

    def easy_process_input(self, key):
        input = self.inputs[key]
        input_prob = utils.symbol_probabilities(input)
        input_entropy = utils.entropy(input_prob.values())

        coded_fse = pyfse.easy_compress(input)
        decoded_fse = pyfse.easy_decompress(coded_fse)

        # Check if the coded stream is within 10% of the entropy
        self.assertLessEqual(len(coded_fse), 1.1 * len(input) * input_entropy / 8)
        self.assertGreaterEqual(len(coded_fse), len(input) * input_entropy / 8)

        self.assertTrue(input == decoded_fse)

        limit = len(input) * input_entropy / 8

        print('Input ({}): {:,} symbols'.format(key, len(input)))
        print('Entropy: {:.2f}'.format(input_entropy))
        print('Theoretical limit: {:,.1f} bytes'.format(limit))
        print('FSE coded stream: {:,} bytes [{:.0f}%]'.format(len(coded_fse), 100 * len(coded_fse) / limit))
        print('Decoding status: {}'.format(input == decoded_fse))

    def test_easy_ascii(self):
        self.easy_process_input('ascii')

    def test_easy_numbers(self):
        self.easy_process_input('numbers')

    def test_easy_binary(self):
        self.easy_process_input('binary')
