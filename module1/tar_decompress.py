import os

folder_path = "./path/to/imagenet-1k/IMAGENET_TRAIN"
target_path = "./path/to/imagenet-1k/train"
tar_list = os.listdir(folder_path)

for file in tar_list:
    확장자 = os.path.splitext(file)[-1]
    if not 확장자 in ['.tar']:
        print(f"file {file} is not selected")
        continue
    file_dir = os.path.join(folder_path, file)
    
    # target_path에 맞는 폴더가 없으면 만들기
    class_name = os.path.splitext(file)[0]
    target_folder_dir = os.path.join(target_path, class_name)
    if not os.path.isdir(target_folder_dir):
        os.mkdir(target_folder_dir)
    
    
    # 압축풀기
    order = f"tar -xvf {file_dir} -C {target_folder_dir}"
    print(os.system(order))
    
    # 압축 푼 파일 삭제하기
    os.remove(file_dir)