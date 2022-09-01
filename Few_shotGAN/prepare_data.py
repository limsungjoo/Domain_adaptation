import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
from torchvision.transforms import InterpolationMode

def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    # img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes=(128, 256, 512, 1024), resample=InterpolationMode.BILINEAR, quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))
    
    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file

    file_name = file.split('/')[-1]
    img = Image.open(file)
    img = img.convert('RGB')
    
    out = resize_multiple(img, sizes=sizes, resample=resample)
    
    return i, out, file_name

def prepare(env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=InterpolationMode.BILINEAR):
    # files = sorted(dataset.imgs, key=lambda x: x[0])
    files = dataset.imgs
    # files = [(i, file) for i, (file, label) in enumerate(files)]
    files = [((i,file), sizes, resample) for i,(file,label) in enumerate(files)]
    total = 0
    
    i_list = np.array([])
    file_n_list = np.array([])
    with multiprocessing.Pool(n_worker) as pool:
        for i, out, file_name in tqdm(pool.starmap(resize_worker, files)):
            print(i, file_name)
            i_list = np.append(i_list,i)
            file_n_list = np.append(file_n_list,file_name)

            for size, img in zip(sizes,out):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1
            if total == 2400:
                break
        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))
    dic = pd.DataFrame({'index':i_list,'file_name':file_n_list})
    dic.to_csv('./target_name.csv',index=None)

# L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])

# def prepare(env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=InterpolationMode.BILINEAR):
#     # k = resize_worker
#     # k_inst, fname = k[:2], k[2]
#     resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

#     # files = sorted(dataset.imgs, key=lambda x: x[0])
#     files = dataset.imgs
#     files = [(i, file) for i, (file, label) in enumerate(files)]
#     total = 0

#     with multiprocessing.Pool(n_worker) as pool:

#         for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
#             for size, img in zip(sizes, imgs):
#                 key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')

#                 with env.begin(write=True) as txn:
#                     txn.put(key, img)

#             total += 1
#             if total == 2400:
#                 break
#         with env.begin(write=True) as txn:
#             txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--size', type=str, default='128,256,512,1024')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--resample', type=str, default='bilinear')
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    
    resample_map = {'lanczos': InterpolationMode.LANCZOS, 'bilinear': InterpolationMode.BILINEAR}

    resample = resample_map[args.resample]
    
    sizes = [int(s.strip()) for s in args.size.split(',')]

    print(f'Make dataset of image sizes:', ', '.join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)
