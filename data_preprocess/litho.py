import numpy as np
import os
import argparse
import tqdm
import glob

from common import read_img_sitk

import cv2 # for resize

def read_multimodal(src_path, dst_path, series):
    for mode in series:
        print('process: ' + mode)
        files = glob.glob("%s/%s/*" % (src_path, mode)) # glob finds all the files
        for f in tqdm.tqdm(files): # twdm shows progress bar
            data = read_img_sitk(f) # simpleITK reads an image and makes it a 512x512 numpy array of float32s
            data = cv2.pyrDown(data, dstsize=(256,256)) # downsample to 256x256
            data = data[None,:,:] # reshape into 1xNxN to work with the rest of FedMedGAN
            name = f.split('/')[-1].replace('.tif', '.npy')
            np.save(dst_path + '/' + mode + '/' + name, data)


def dataset_preprocess(src_path, dst_path):
    series = ['A', 'B']
    assert os.path.exists(src_path), "Source path doesn't exist!"
    for mode in series:
        assert os.path.exists(src_path + '/' + mode), "Source path doesn't have A and B subfolders!"
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for mode in series:
        if not os.path.exists(dst_path + '/' + mode):
            os.mkdir(dst_path + '/' + mode)
    read_multimodal(src_path, dst_path, series)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FedMedGAN preprocessing for lithography step")
    parser.add_argument("--data_path", default="/home/dy245/Data/PreEtch_9wafer_TGAP_ZiwangPairs", nargs='?', type=str, help="path to input image data")
    parser.add_argument("--generated_path", default="/home/dy245/Data/TGAP_preprocessed", nargs='?', type=str, help="path to output npy files")
    args = parser.parse_args()
    dataset_preprocess(src_path=args.data_path, dst_path=args.generated_path)