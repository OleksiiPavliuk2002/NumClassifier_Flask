import os
from shutil import copyfile
import re
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.join(BASE_DIR, 'images', 'custom')
os.makedirs(CUSTOM_DIR, exist_ok=True)

MNIST_DIR = os.path.join(BASE_DIR, 'images')

def extract_index_from_name(name):
    m = re.search(r'mnist-(\d+)\.png$', name)
    if m:
        return int(m.group(1))
    return None

def build_custom_digits():
    # read test labels
    import gzip
    import numpy as np
    lab_path = os.path.join(BASE_DIR, 'mnist', 't10k-labels-idx1-ubyte.gz')
    with gzip.open(lab_path, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    files = sorted([f for f in os.listdir(MNIST_DIR) if f.startswith('mnist-') and f.endswith('.png')])
    seen = {}
    for fname in files:
        idx = extract_index_from_name(fname)
        if idx is None:
            continue
        label = int(labels[idx-1])
        if label not in seen:
            src = os.path.join(MNIST_DIR, fname)
            dst = os.path.join(CUSTOM_DIR, f'digit_{label}.png')
            copyfile(src, dst)
            seen[label] = dst
        if len(seen) >= 10:
            break

    # ensure we have 10 images
    for d in range(10):
        path = os.path.join(CUSTOM_DIR, f'digit_{d}.png')
        if not os.path.exists(path):
            # fallback: create a blank image
            Image.new('L', (28,28), color=0).save(path)

    print('Created/updated 10 digit images in', CUSTOM_DIR)


if __name__ == '__main__':
    build_custom_digits()
