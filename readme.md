# For quickly processing radio imaging data
(sacrificing some quality)

## GPU-imager

Use cupy to run Cuda C code (src/funcGridding.cu) for the gridding and use cufft to do the 2D-FFT. In the end we can perform imaging for 100 frame within 0.05s, but quality compromises.

## Autoencoder

In autoencoderhandler.py, the encoder-bottleneck-decoder structure, aming to use small set of numer (e.g. 256 float32) to represent a image.