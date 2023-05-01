# CONVNEXT

Github에 repo 만들어야함

# 모델설명

2020년 ViT가 등장하며 이미지 관련 task에서 transformer가 가장 지배적인 backbone architecture가 되었다. ViT의 등장 이후, convolution이 가지고 있는 inductive bias를 transformer에 추가하고자 하였고, 그 결과 sliding window 전략을 사용하는 Hierarchical vision transformer인 Swin Transformer가 소개되었다.  
이런 상황에서 이 모델은 ResNet에서 transformer가 갖는 design decision을 도입해서 convolution network의 성능을 transformer network와 비등하게 만든 모델이다.

ConvNext에서 적용한 것들은 다음과 같다.
1. transformer의 training techniques 적용
2. Convolution network의 stage compute ratio를 변경
3. Convolution network의 stem부분을 patchify로 변경
4. ResNeXt의 depthwise convolution을 적용
5. Inverted bottleneck구조 적용
6. 커널 사이즈 변경
7. Activation Function을 ReLU대신 GELU로 변경
8. Batch Normalization을 Layer Normalization으로 변경
9. Downsampling Layer 분리

각 변경사항별 성능향상도는 아래와 같다.

|변경사항|변정전||변경후|변화량|
|:---:|:---:|---|:---:|:---:|
|Training Technique|76.1%|▶|78.8%|+2.7%|
|stage compute ratio|78.8%|▶|79.4%|+0.6%|
|stem 변경|79.4%|▶|79.5%|+0.1%|
|ResNeXt-ify(Depth Conv적용)|79.5%|▶|80.5%|+1.0%|
|depthwise conv 적용|79.5%|▶|80.5%|+1.0%|
|inverted bottlenet 적용|80.5%|▶|80.6%|+0.1%|
|large-kernel conv layer 위로 올리기|80.6%|▶|79.9%|-0.5%|
|GELU 적용|80.6%|▶|80.6%|-|
|Activation function 줄이기|80.6%|▶|81.3%|+0.7%|
|Normalization layer 줄이기|81.3%|▶|81.4%|+0.1%|
|BN을 LN으로 변경|81.4%|▶|81.5%|+0.1%|
|downsampling layer 분리|81.5%|▶|82.0%|+0.5%|

### 1. Transformer의 training techniques 적용
- 훈련 epochs를 300으로 증가
- AdamW optimizer 적용
- Mixup, Cutmix, RandArgument, RandomErasing등 Data augmentation 적용
- Stochastic Depth, Label Smoothing등의 정규화 schemes 적용

### 2. Convolution network의 stage compute ratio를 변경

Swin-T에서 stage ratio는 1:1:3:1이다. (Large Swin Transformer의 경우 1:1:9:1)
따라서 이를 모방해 ResNet의 stage ratio를 (3,3,9,3)으로 변경하였다.

### 3. Convolution network의 stem부분을 patchify로 변경

ResNet에서는 7x7 커널을 가진 stride2인 convolution을 사용하고 maxpool을 사용하지만, ConvNeXt에서는 stem cell을 4x4 kernel을 갖는 stride 4 convolution으로 변경하였다.

### 4. ResNeXt의 depthwise convolution을 적용

depthwise convolution을 사용하였다. Network의 width는 Swin-T의 채널수와 동일한 96으로 증가시켰다.

### 5. Inverted bottleneck 구조 적용

Transformer에서 중요한 디자인 요소 중 하나인 inverted bottleneck을 적용하였다.

### 6. 커널 사이즈 변경

Depthwise convolution layer를 먼저 수행하고, 효율적이고 dense한 1x1 convolution은 그 이후에 수행하도록 하였다.
convolution block
1. 7x7 depthwise convolution 96 -> 96
2. 1x1 convolution 96 -> 384
3. 1x1 convolution 384 -> 96

### 7. Activation Function을 ReLU대신 GELU로 변경

Activation function을 transformer에서 사용하는 GELU를 사용하였다.
그리고 transformer에서는 convolution에서보다 더 적은 activation function을 사용하므로 이를 차용하여 convolution block에서 두 1x1 convolution 사이에만 activation function을 적용하였다.

### 8. Batch Normalization을 Layer Normalization으로 변경

Batch Normalization 대신에 Layer Normalization을 사용하였다.

### 9. Downsampling Layer 분리

Stage 사이에 2x2 kernel을 갖고, stride가 2인 convolution을 이용하여 downsampling을 적용한다.

### 전체적인 모델의 architecture

|stage|block 구조|block 개수|output size|
|:---:|:---:|:---:|:---:|
|stem|4x4, 96, stride 4|1|56x56|
|res2|depthwise conv 7x7, 96</br>1x1 conv, 384</br>1x1 conv, 96|3|56x56|
|res3|depthwise conv 7x7, 192</br>1x1 conv, 768</br>1x1 conv, 192|3|28x28|
|res4|depthwise conv 7x7, 384</br>1x1 conv, 1536</br>1x1 conv, 384|9|14x14|
|res5|depthwise conv 7x7, 768</br>1x1 conv, 3072</br>1x1 conv, 768|3|7x7|
|FC|클래스 수로 줄이는 MLP|1|1000/21000

# Requirements 설명
> 필요시 python library 및 version 설명해야함

필요한 library
- torch >= 1.8.0
- torchvision >= 0.9.0
- timm >= 0.3.2


# 실행방법

모델에 대한 디테일은 models파일에  
convnext_isotropic.py와 convnext.py를 통해서 확인할 수 있다.
convnext.py는 일반적인 ConvNeXt 모델을 뜻하고,  
convnext_isotropic.py는 ViT스타일로 일반화시킨 모델로, downsampling을 없애고 feature resolution을 고정한 모델을 뜻한다.

## pretrained model 다운로드
pretrained model
### 1. pretrained on IMAGENET 1k

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt-T | 224x224 | 82.1 | 28M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |
| ConvNeXt-S | 224x224 | 83.1 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| ConvNeXt-B | 224x224 | 83.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |
| ConvNeXt-B | 384x384 | 85.1 | 89M | 45.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth) |
| ConvNeXt-L | 224x224 | 84.3 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth) |
| ConvNeXt-L | 384x384 | 85.5 | 198M | 101.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth) |

### 2. pretrained on IMAGENET 22k
IMAGENET 22k에서 학습된 모델과 IMAGENET 22k에서 학습되고 IMAGENET 1k에서 Fine-tuned된 모델

| name | resolution |acc@1 | #params | FLOPs | 22k model | finetuned on 1k model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
| ConvNeXt-T | 224x224 | 82.9 | 29M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth)   | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth)
| ConvNeXt-T | 384x384 | 84.1 | 29M | 13.1G |     -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth)
| ConvNeXt-S | 224x224 | 84.6 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth)   | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth)
| ConvNeXt-S | 384x384 | 85.8 | 50M | 25.5G |     -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth)
| ConvNeXt-B | 224x224 | 85.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth)   | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth)
| ConvNeXt-B | 384x384 | 86.8 | 89M | 47.0G |     -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth)
| ConvNeXt-L | 224x224 | 86.6 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth)  | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth)
| ConvNeXt-L | 384x384 | 87.5 | 198M | 101.0G |    -         | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth)
| ConvNeXt-XL | 224x224 | 87.0 | 350M | 60.9G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth) | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth)
| ConvNeXt-XL | 384x384 | 87.8 | 350M | 179.0G |  -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth)

### 3. pretrained on IMAGENET 1k (isotropic model)

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt-S | 224x224 | 78.7 | 22M | 4.3G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth) |
| ConvNeXt-B | 224x224 | 82.0 | 87M | 16.9G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth) |
| ConvNeXt-L | 224x224 | 82.6 | 306M | 59.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_iso_large_1k_224_ema.pth) |

## 이미지에 대한 classification 수행방법

image 폴더에 classification 수행할 모든 이미지 파일을 저장

| model description | image size | model name | args |
|:---:|:---:|:---:|:---:|:---:|
| trained on 22k fine tuned on 1k | 224x224 | ConvNeXt-T | convnext_tiny_in22ft1k |
| trained on 22k fine tuned on 1k | 224x224 | ConvNeXt-S | convnext_small_in22ft1k |
| trained on 22k fine tuned on 1k | 224x224 | ConvNeXt-B | convnext_base_in22ft1k |
| trained on 22k fine tuned on 1k | 224x224 | ConvNeXt-L | convnext_large_in22ft1k |
| trained on 22k fine tuned on 1k | 224x224 | ConvNeXt-XL | convnext_xlarge_in22ft1k |
| trained on 22k fine tuned on 1k | 384x384 | ConvNeXt-T | convnext_tiny_384_in22ft1k |
| trained on 22k fine tuned on 1k | 384x384 | ConvNeXt-S | convnext_small_384_in22ft1k |
| trained on 22k fine tuned on 1k | 384x384 | ConvNeXt-B | convnext_base_384_in22ft1k |
| trained on 22k fine tuned on 1k | 384x384 | ConvNeXt-L | convnext_large_384_in22ft1k |
| trained on 22k fine tuned on 1k | 384x384 | ConvNeXt-XL | convnext_xlarge_384_in22ft1k |
| trained on 22k | 224x224 | ConvNeXt-T | convnext_tiny_in22k |
| trained on 22k | 224x224 | ConvNeXt-S | convnext_small_in22k |
| trained on 22k | 224x224 | ConvNeXt-B | convnext_base_in22k |
| trained on 22k | 224x224 | ConvNeXt-L | convnext_large_in22k |
| trained on 22k | 224x224 | ConvNeXt-XL | convnext_xlarge_in22k |

아래 명령어 수행

```
python classification.py --model convnext_tiny_in22
```

matplotlib을 통하여 각 이미지에 대한 출력을 얻을 수 있다.


## detection 수행방법


# model pseudo code
너무 간단하게 쓰면 안됨, 자세하게 썼다가 틀려도 안됨

# ㅁㄴㅇㄹasdfㅁㄴㅇㄹㅁㄴㅇㄹasdf