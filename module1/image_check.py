from PIL import Image
import os
import numpy as np
import string

# train 폴더에서 깨진 파일이 있다면 다 모아서 마지막에 알려주는 역할을 한다.

root_dir = os.path.join("./path/to/imagenet-1k/train")
checkdir = os.listdir(root_dir)
format = [".jpg", ".jpeg", "JPEG", "JPG"]
n=1
truncated_file_location = []
folders = []

for folder_name in checkdir:
    print(f'{n}th folder_name:{folder_name}')
    n+=1
    for(path, dirs, f) in os.walk(os.path.join(root_dir, folder_name)):
        for file in f:
            if file.endswith(tuple(format)):
                try:
                    image = Image.open(path+"/"+file)
                    img = np.asarray(image)
                    # print(image)
                except Exception as e:
                    print("An exception is raised:", e)
                    print(file)
                    truncated_file_location.append([path, file])
                    folders.append(path.split('/')[-1])

for item in truncated_file_location:
    print(item)
    
folders.sort()
for item in folders:
    print(item)
    
# folder_list = os.listdir(root_dir)
# folder_list.sort()
# for fff in folder_list:
#     print(fff)