import os
from PIL import Image
from resizeimage import resizeimage
from progressbar import ProgressBar

def resize(infolder, outfolder, imagefile, size):
    """
        infolder: folder that input image lives
        outfolder: folder that output image will live
        imagefile: image filename, like img.png
        size: list: [150,150]
    """
    fd_img = open(os.path.join(infolder,imagefile),'r')
    img = Image.open(fd_img)
    img = resizeimage.resize_crop(img, size)
    img.save(os.path.join(outfolder,imagefile), img.format)
    fd_img.close()

def resize_in_folder(folder,size):
    for img in os.listdir(folder):
        parent = os.path.dirname(folder)
        outfolder = os.path.join(parent,os.path.basename(folder) + '_resize')
        if not os.path.exists(outfolder): os.mkdir(outfolder)
        resize(folder, outfolder, img, size)

def resize_kitti():
    settings = ['road', 'campus', 'residential', 'city']
    for setting in settings:
        root = '/om/data/public/mbchang/udcign-data/kitti/raw/videos/' + setting + '/'
        ch = 'image_02'  # color
        size = [150,150]
        pbar = ProgressBar().start()
        i = 0
        for folder in os.listdir(root):
            img_folder = os.path.join(root, folder + '/' + ch + '/data')
            resize_in_folder(img_folder, size)

            pbar.update(i + 1)
            i += 1
        pbar.finish()

def resize_toyota():
    root = '/om/data/public/mbchang/udcign-data/toyota/pics'
    size = [150,150]
    pbar = ProgressBar().start()
    i = 0
    for folder in os.listdir(root):
        resize_in_folder(os.path.join(root,folder), size)
        pbar.update(i + 1)
        i += 1
    pbar.finish()

if __name__ == '__main__':
    resize_toyota()
