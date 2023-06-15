import os
from PIL import Image
import matplotlib.pyplot as plt

raw_im_path = './path/to/imagenet-1k/train'
target_path = './path/to/imagenet-1k/train_50'

print(os.listdir(raw_im_path))

for path, dirs, files in (os.walk(raw_im_path)):
    for class_dir in dirs:
        target_folder = os.path.join(target_path, class_dir)
        print("working on ", class_dir)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        for class_path, img_dir, file_dirs in (os.walk(os.path.join(path, class_dir))):
            for file_dir in file_dirs:
                target_file_dir = os.path.join(target_folder, file_dir)
                if os.path.exists(target_file_dir):
                    pass
                im = Image.open(os.path.join(class_path, file_dir))
                im = im.convert('RGB')
                im.save(target_file_dir, quality=50)