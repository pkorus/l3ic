## Introduction

...

## Architecture at a Glance

- We used the auto-encoder architecture from [Liu et al. CVPR Workshops'18](https://arxiv.org/abs/1806.01496)
- We implemented our own quantizer and entropy estimator
- We don't use any normalizations (e.g., GDNs) & regularize only based on the entropy of the quantized latent representation
- The deployed model uses a code-book with 32 values (integers from -15 to 16)
- We used a state-of-the-art entropy codec ([FSE](https://github.com/Cyan4973/FiniteStateEntropy))
- The latent representation is entropy-coded channel-wise - this brings savings as the images get larger and allows for random access to individual channels (may be useful e.g., when [doing vision directly on the latent representation](https://arxiv.org/abs/1803.06131))
- We control image quality using the number of latent channels (3 models with 16, 32 and 64 channels are provided)

More details available [here](...)

## Installation

```bash
> git clone https://github.com/pkorus/neural-image-compression
> cd neural-image-compression
> pip3 install -r requirements.txt
> git submodule init
> git submodule update
> cd pyfse
> make
> cd ..
> python3 demo.py -m 8k
```

## Quality Benchmarks



## Speed Benchmarks

TLDR:
- With mid-range server hardware (CPU/GPU) compressing a 1920 x 1080 RGB image takes 0.19 s, 
- the corresponding decompression takes 0.17 s.

The code has not been particularly optimized, but still delivers competitive processing speed. It is significantly faster than some of the other models you can find online (e.g., [Mentzer et al. CVPR'18](https://github.com/fab-jul/imgcomp-cvpr) which is reported to require ~6 min for fairly small 768 x 512 Kodak images). With modern GPUs, the entropy codec starts to become the bottleneck - and we used one of the fastest things out there (the [FSE codec](https://github.com/Cyan4973/FiniteStateEntropy) from Yann Collet based on [asymmetric numeral systems](https://arxiv.org/abs/1311.2540) from Jarek Duda).

As a general reference, encoding a 1920 x 1080 px image using standard codecs takes approx. (n i7-7700 CPU @ 3.60GHz):
- JPEG with 1 thread takes between 0.061s (Q=30) and 0.075 s (Q=90) [inclusive of writing time to RAM disk; *pillow* library]
- JPEG 2000 with 1 thread takes ~0.61 s regardless of the quality level [inclusive of writing time to RAM disk; *glymur* library]
- BPG with 4 parallel threads takes 2.4 s (Q=1), 1.25 s (Q=20) and 0.72 s (Q=30) [inclusive of PNG reading time from RAM disk; *bpgenc* command line tool].

**Average processing time [sec] for 512 x 512 images (8k model)**

| CPU/GPU                       | DCN Encoding | DCN Decoding | AFI Encoding | AFI Decoding |
|-------------------------------|--------------|--------------|--------------|--------------|
| i7-7700 CPU @ 3.60GHz         | 0.2165       | 0.3330       | 0.2272       | 0.3317       |
| K80 + Xeon E5-2680 @ 2.40GHz  | 0.0250       | 0.0316       | 0.0570       | 0.0492       |
| P40 + Xeon E5-2680 @ 2.40GHz  | 0.0093       | 0.0160       | 0.0720       | 0.0375       |
| V100 + Xeon E5-2680 @ 2.40GHz |              |              |              |              |

**Average processing time [sec] for 1920 x 1080 images (8k model)**

| CPU/GPU                       | DCN Encoding | DCN Decoding | AFI Encoding | AFI Decoding |
|-------------------------------|--------------|--------------|--------------|--------------|
| i7-7700 CPU @ 3.60GHz         | 1.8052       | 2.7678       | 1.8753       | 2.7901       |
| K80 + Xeon E5-2680 @ 2.40GHz  | 0.1944       | 0.2622       | 0.3335       | 0.3360       |
| P40 + Xeon E5-2680 @ 2.40GHz  | 0.0558       | 0.1123       | 0.1895       | 0.1684       |
| V100 + Xeon E5-2680 @ 2.40GHz |              |              |              |              |
