import sys

import matplotlib.pyplot as plt
import PIL
from PIL import Image
import json
import argparse
from pathlib import Path
import os

import torch
import torchvision
import torchvision.transforms as T

from timm import create_model

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt inference for image classification', add_help=False)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--model', default='convnext_tiny_in22k',
                        help='convnext model')
    
    return parser

def main(args):
    model_name = args.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device = ", device)
    # create a ConvNeXt model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
    model = create_model(model_name, pretrained=True).to(device)

    # Define transforms for test
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
    NORMALIZE_STD = IMAGENET_DEFAULT_STD
    SIZE = 256

    # Here we resize smaller edge to 256, no center cropping
    transforms = [
                T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
                ]

    transforms = T.Compose(transforms)

    # labels to word를 다운로드받아야 나중에 결과가 어떤건지 알 수 있겠네;;
    imagenet_labels = json.load(open('label_to_words.json'))

    # image 폴더 내 이미지 불러오기
    image_folder = './image/'
    images_dir = [os.path.join(root, file) for root, dirs, files in os.walk(image_folder) for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_dir in images_dir:
        img = PIL.Image.open(img_dir)
        img_tensor = transforms(img).unsqueeze(0).to(device)

        # inference
        output = torch.softmax(model(img_tensor), dim=1)
        top5 = torch.topk(output, k=5)
        top5_prob = top5.values[0]
        top5_indices = top5.indices[0]

        for i in range(5):
            labels = imagenet_labels[str(int(top5_indices[i]))]
            prob = "{:.2f}%".format(float(top5_prob[i])*100)
            print(labels, prob)
        print("========")

        plt.imshow(img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)