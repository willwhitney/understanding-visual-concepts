import cv2
import os
import numpy as np
from images2gif import writeGif
from PIL import Image
import os
import pprint

# Usage: Download images2gif
# Change
#   for im in images:
#       palettes.append( getheader(im)[1] )
# To
#   for im in images:
#       palettes.append(im.palette.getdata()[1])

def create_gif(images_root):
    """
        writeGif(filename, images, duration=0.1, loops=0, dither=1)
            Write an animated gif from the specified images.
            images should be a list of numpy arrays of PIL images.
            Numpy images of type float should have pixels between 0 and 1.
            Numpy images of other types are expected to have values between 0 and 255.
    """
    def img_id(filename):
        begin = len('changing_')
        end = filename.find('_amount')
        return int(filename[begin:end])

    file_names = sorted([fn for fn in os.listdir(images_root) if fn.endswith('.png')], key=lambda x: img_id(x))
    images = [Image.open(os.path.join(images_root,fn)) for fn in file_names]
    filename = os.path.join(images_root, "gif.GIF")
    # print filename
    writeGif(filename, images, duration=0.05)

if __name__ == '__main__':
    root = '/Users/MichaelChang/Dropbox (MIT Solar Car Team)/MacHD/Documents/Important/MIT/Research/SuperUROP/Code/unsupervised-dcign/renderings/mutation'
    images_root = 'ballsdubhead_Mar_03_19_11'
    for exp in [f for f in os.listdir(os.path.join(root,images_root)) if '.txt' not in f and not f.startswith('.')]:
        for demo in [f for f in os.listdir(os.path.join(*[root,images_root,exp])) if not f.startswith('.')]:
            print demo
            create_gif(os.path.join(*[root, images_root, exp, demo]))
