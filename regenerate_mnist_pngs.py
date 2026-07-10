import gzip
import os
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MNIST_DIR = os.path.join(BASE_DIR, 'mnist')
OUT_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(OUT_DIR, exist_ok=True)

with gzip.open(os.path.join(MNIST_DIR, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
    f.read(16)
    buf = f.read()
    X_test = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 784)

for i, img in enumerate(X_test, start=1):
    arr = img.reshape(28, 28).astype('uint8')
    im = Image.fromarray(arr, mode='L')
    im.save(os.path.join(OUT_DIR, f'mnist-{i:06d}.png'))

print(f'Wrote {len(X_test)} images to {OUT_DIR}')
