import os
import shutil

# 이미지넷 데이터셋 경로
dataset_path = 'path/to/imagenet-1k/val'

# 클래스 이름과 클래스 ID가 저장된 텍스트 파일 경로
classes_file = 'path/to/imagenet-1k/ILSVRC2012_validation_ground_truth.txt'

# 텍스트 파일을 불러와서 1번부터 각 번호가 몇번째 class인지 알려줌
with open(classes_file, 'r') as f:
    idx = 1
    classes = []
    for line in f:
        classes.append(line.strip().split(',')[0])
        # class_id = idx
        # class_name = line.strip().split(',')[0]
        # classes[class_name] = idx
        idx += 1

# indices = [i for i in range(len(classes)) if classes[i] == '318']

# 클래스별 폴더 생성 폴더 이름은 클래스 번호로
for class_id in classes:
    class_folder = os.path.join(dataset_path, class_id)
    os.makedirs(class_folder, exist_ok=True)

# 각 이미지 파일을 클래스별 폴더로 이동
for image_file in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_file)
    if os.path.isfile(image_path):
        class_id = image_file.split('_')[2]
        idx = int(class_id) - 1
        class_folder = os.path.join(dataset_path, classes[idx])
        shutil.move(image_path, class_folder)
