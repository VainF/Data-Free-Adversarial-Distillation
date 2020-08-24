import os, shutil
import random
random.seed(1)
from PIL import Image

base_path = os.getcwd()

data_path = 'caltech101/101_ObjectCategories'
categories = os.listdir(data_path)
os.makedirs( 'caltech101/101_ObjectCategories_split', exist_ok=True)
os.makedirs( 'caltech101/101_ObjectCategories_split/train', exist_ok=True )
os.makedirs( 'caltech101/101_ObjectCategories_split/test', exist_ok=True )

train_path = os.path.join(base_path, "caltech101/101_ObjectCategories_split/train") 
test_path = os.path.join(base_path, "caltech101/101_ObjectCategories_split/test")
for cat in categories:
    image_files = os.listdir(os.path.join(data_path, cat))
    num_choices = max( int(0.2*len(image_files)), 2 )
    test_idx = random.sample( list(range(len(image_files))), k=num_choices )

    for idx, imgf in enumerate(image_files):
        origin_path = os.path.join(data_path, cat,  imgf)
        if idx not in test_idx:
            dst_root = train_path
        else:
            dst_root = test_path

        dest_dir = os.path.join(dst_root, cat)
        dest_path = os.path.join(dst_root, cat, imgf)

        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        shutil.copy2(origin_path, dest_path)
