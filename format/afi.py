import io
import json
import numpy as np
from scipy import cluster
from collections import Counter
from pathlib import Path

from scipy.cluster.vq import vq
from skimage.measure import compare_ssim, compare_psnr

from pyfse import pyfse
from format import utils


class AFIError(Exception):
    pass


def simulate_compression(dcn, batch_x):
    """
    Simulate AFI compression and return decompressed image and byte count.
    """

    # Compress each image
    compressed_image = compress(dcn, batch_x)
    batch_y = decompress(dcn, compressed_image)

    return batch_y, len(compressed_image)


def compress_n_stats(dcn, batch_x):

    batch_y = np.zeros_like(batch_x)
    stats = {
        'ssim': np.zeros((batch_x.shape[0])),
        'psnr': np.zeros((batch_x.shape[0])),
        'entropy': np.zeros((batch_x.shape[0])),
        'bytes': np.zeros((batch_x.shape[0])),
        'bpp': np.zeros((batch_x.shape[0]))
    }

    for image_id in range(batch_x.shape[0]):
        batch_y[image_id], image_bytes = simulate_compression(dcn, batch_x[image_id:image_id + 1])
        batch_z = dcn.compress(batch_x[image_id:image_id + 1])
        stats['bytes'][image_id] = image_bytes
        stats['entropy'][image_id] = utils.entropy(batch_z, dcn.get_codebook())
        stats['ssim'][image_id] = compare_ssim(batch_x[image_id], batch_y[image_id], multichannel=True, data_range=1)
        stats['psnr'][image_id] = compare_psnr(batch_x[image_id], batch_y[image_id], data_range=1)
        stats['bpp'][image_id] = 8 * image_bytes / batch_x[image_id].shape[0] / batch_x[image_id].shape[1]

    if batch_x.shape[0] == 1:
        for k in stats.keys():
            stats[k] = stats[k][0]

    return batch_y, stats


def compress(model, batch_x, verbose=False):
    """
    Serialize the image as a bytes sequence. The feature maps are encoded as separate layers.

    ## Analysis Friendly Image (AFI) File structure:

    - Latent shape H x W x N = 3 x 1 byte (uint8)
    - Length of coded layer sizes = 2 bytes (uint16)
    - Coded layer sizes:
        - FSE encoded uint16 array of size 2 * N bytes (if possible to compress)
        - ...or RAW bytes
    - Coded layers:
        - FSE encoded uint8 array of latent vector size
        - ...or RLE encoded uint16 (number) + uint8 (byte) if all bytes are the same

    """

    if batch_x.ndim == 3:
        batch_x = np.expand_dims(batch_x, axis=0)

    assert batch_x.ndim == 4
    assert batch_x.shape[0] == 1

    image_stream = io.BytesIO()

    # Get latent space representation
    batch_z = model.compress(batch_x)
    latent_shape = np.array(batch_z.shape[1:], dtype=np.uint8)

    # Write latent space shape to the bytestream
    image_stream.write(latent_shape.tobytes())

    # Encode feature layers separately
    coded_layers = []
    code_book = model.get_codebook()
    if verbose:
        print('[AFI Encoder]', 'Code book:', code_book)

    if len(code_book) > 256:
        raise AFIError('Code-books with more than 256 centers are not supported')

    for n in range(latent_shape[-1]):
        # TODO Should a code book always be used? What about integers?
        indices, _ = cluster.vq.vq(batch_z[:, :, :, n].reshape((-1)), code_book)

        try:
            # Compress layer with FSE
            coded_layer = pyfse.easy_compress(bytes(indices.astype(np.uint8)))
        except pyfse.FSESymbolRepetitionError:
            # All bytes are identical, fallback to RLE
            coded_layer = np.uint16(len(indices)).tobytes() + np.uint8(indices[0]).tobytes()
        except pyfse.FSENotCompressibleError:
            # Stream does not compress
            coded_layer = np.uint8(indices).tobytes()
        finally:
            if len(coded_layer) == 1:
                if verbose:
                    layer_stats = Counter(batch_z[:, :, :, n].reshape((-1))).items()
                    print('[AFI Encoder]', 'Layer {} values:'.format(n), batch_z[:, :, :, n].reshape((-1)))
                    print('[AFI Encoder]', 'Layer {} code-book indices:'.format(n), indices.reshape((-1))[:20])
                    print('[AFI Encoder]', 'Layer {} hist:'.format(n), layer_stats)

                raise AFIError('Layer {} data compresses to a single byte? Something is wrong!'.format(n))
            coded_layers.append(coded_layer)

    # Show example layer
    if verbose:
        n = 0
        layer_stats = Counter(batch_z[:, :, :, n].reshape((-1))).items()
        print('[AFI Encoder]', 'Layer {} values:'.format(n), batch_z[:, :, :, n].reshape((-1)))
        print('[AFI Encoder]', 'Layer {} code-book indices:'.format(n), indices.reshape((-1))[:20])
        print('[AFI Encoder]', 'Layer {} hist:'.format(n), layer_stats)

    # Write the layer size array
    layer_lengths = np.array([len(x) for x in coded_layers], dtype=np.uint16)

    try:
        coded_lengths = pyfse.easy_compress(layer_lengths.tobytes())
        if verbose: print('[AFI Encoder]', 'FSE coded lengths')
    except pyfse.FSENotCompressibleError:
        # If the FSE coded stream is empty - it is not compressible - save natively
        if verbose: print('[AFI Encoder]', 'RAW coded lengths')
        coded_lengths = layer_lengths.tobytes()

    if verbose:
        print('[AFI Encoder]', 'Coded lengths #', len(coded_lengths), '=', coded_lengths)
        print('[AFI Encoder]', 'Layer lengths = ', layer_lengths)

    if len(coded_lengths) == 0:
        raise RuntimeError('Empty coded layer lengths!')

    image_stream.write(np.uint16(len(coded_lengths)).tobytes())
    image_stream.write(coded_lengths)

    # Write individual layers
    for layer in coded_layers:
        image_stream.write(layer)

    return image_stream.getvalue()


def decompress(model, stream, verbose=False):
    """
    Deserialize an image from the given bytes sequence. See docs of compress for stream details.
    """

    if type(stream) is bytes:
        stream = io.BytesIO(stream)
    elif type(stream) is io.BytesIO:
        pass
    elif not hasattr(stream, 'read'):
        raise ValueError('Unsupported stream type!')

    # Read the shape of the latent representation
    latent_x, latent_y, n_latent = np.frombuffer(stream.read(3), np.uint8)

    code_book = model.get_codebook()
    # Read the array with layer sizes
    layer_bytes = np.frombuffer(stream.read(2), np.uint16)
    coded_layer_lengths = stream.read(int(layer_bytes))

    if verbose:
        print('[AFI Decoder]', 'Latent space', latent_x, latent_y, n_latent)
        print('[AFI Decoder]', 'Layer bytes', layer_bytes)

    if layer_bytes != 2 * n_latent:
        if verbose:
            print('[AFI Decoder]', 'Decoding FSE L')
            print('[AFI Decoder]', 'Decoding from', coded_layer_lengths)
        layer_lengths_bytes = pyfse.easy_decompress(coded_layer_lengths)
        layer_lengths = np.frombuffer(layer_lengths_bytes, dtype=np.uint16)
    else:
        if verbose:
            print('[AFI Decoder]', 'Decoding RAW L')
        layer_lengths = np.frombuffer(coded_layer_lengths, dtype=np.uint16)

    if verbose:
        print('[AFI Decoder]', 'Layer lengths', layer_lengths)

    # Create the latent space array
    batch_z = np.zeros((1, latent_x, latent_y, n_latent))

    # Decompress the features separately
    for n in range(n_latent):
        coded_layer = stream.read(int(layer_lengths[n]))
        try:
            if len(coded_layer) == 3:
                # RLE encoding
                count = np.frombuffer(coded_layer[:2], dtype=np.uint16)[0]
                layer_data = coded_layer[-1:] * int(count)
            elif len(coded_layer) == int(latent_x) * int(latent_y):
                # If the data could not have been compressed, just read the raw stream
                layer_data = coded_layer
            else:
                layer_data = pyfse.easy_decompress(coded_layer, 4 * latent_x * latent_y)
        except pyfse.FSEException as e:
            print('[AFI Decoder]', 'ERROR while decoding layer', n)
            print('[AFI Decoder]', 'Stream of size', len(coded_layer), 'bytes =', coded_layer)
            raise e
        batch_z[0, :, :, n] = code_book[np.frombuffer(layer_data, np.uint8)].reshape((latent_x, latent_y))

    # Show example layer
    if verbose:
        n = 0
        layer_stats = Counter(batch_z[:, :, :, n].reshape((-1))).items()
        print('[AFI Decoder]', 'Layer {} values:'.format(n), batch_z[:, :, :, n].reshape((-1)))
        # print('[AFI Encoder]', 'Layer {} code-book indices:'.format(n), indices.reshape((-1))[:20])
        print('[AFI Decoder]', 'Layer {} hist:'.format(n), layer_stats)

    # Use the DCN decoder to decompress the RGB image
    return model.decompress(batch_z)


def global_compress(dcn, batch_x):
    # Naive FSE compression of the entire latent repr.
    batch_z = dcn.compress(batch_x)
    indices, distortion = vq(batch_z.reshape((-1)), dcn.get_codebook())
    return pyfse.easy_compress(bytes(indices.astype(np.uint8)))