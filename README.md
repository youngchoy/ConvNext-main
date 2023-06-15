# CONVNEXT v1

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

#### 1. Transformer의 training techniques 적용
- 훈련 epochs를 300으로 증가
- AdamW optimizer 적용
- Mixup, Cutmix, RandArgument, RandomErasing등 Data augmentation 적용
- Stochastic Depth, Label Smoothing등의 정규화 schemes 적용

#### 2. Convolution network의 stage compute ratio를 변경

Swin-T에서 stage ratio는 1:1:3:1이다. (Large Swin Transformer의 경우 1:1:9:1)
따라서 이를 모방해 ResNet의 stage ratio를 (3,3,9,3)으로 변경하였다.

#### 3. Convolution network의 stem부분을 patchify로 변경

ResNet에서는 7x7 커널을 가진 stride2인 convolution을 사용하고 maxpool을 사용하지만, ConvNeXt에서는 stem cell을 4x4 kernel을 갖는 stride 4 convolution으로 변경하였다.

#### 4. ResNeXt의 depthwise convolution을 적용

depthwise convolution을 사용하였다. Network의 width는 Swin-T의 채널수와 동일한 96으로 증가시켰다.

#### 5. Inverted bottleneck 구조 적용

Transformer에서 중요한 디자인 요소 중 하나인 inverted bottleneck을 적용하였다.

#### 6. 커널 사이즈 변경

Depthwise convolution layer를 먼저 수행하고, 효율적이고 dense한 1x1 convolution은 그 이후에 수행하도록 하였다.
convolution block
1. 7x7 depthwise convolution 96 -> 96
2. 1x1 convolution 96 -> 384
3. 1x1 convolution 384 -> 96

#### 7. Activation Function을 ReLU대신 GELU로 변경

Activation function을 transformer에서 사용하는 GELU를 사용하였다.
그리고 transformer에서는 convolution에서보다 더 적은 activation function을 사용하므로 이를 차용하여 convolution block에서 두 1x1 convolution 사이에만 activation function을 적용하였다.

#### 8. Batch Normalization을 Layer Normalization으로 변경

Batch Normalization 대신에 Layer Normalization을 사용하였다.

#### 9. Downsampling Layer 분리

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


# model pseudo code

ConvNeXt-T의 pseudo code는 아래와 같다.
<img src="https://github.com/youngchoy/ConvNext-main/assets/77494237/7424d51e-f11f-4696-9ed0-f3f0dae344f6">


# Requirements 설명

## 학습시 필요한 library
- 파이썬 3.9.0 사용
- torch == 1.8.1
- torchvision == 0.9.0
torch와 torchvision은 아래의 명령어를 이용하여 다운로드 받을 수 있다.
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
- timm == 0.9.2
- tensorboardX == 2.6

## Inference시 필요한 library
- 파이썬 3.9.0 사용
- torch > 2.0

# 실행방법

모델에 대한 디테일은 models파일의  
convnext_isotropic.py와 convnext.py를 통해서 확인할 수 있다.
convnext.py는 일반적인 ConvNeXt 모델을 뜻하고,  
convnext_isotropic.py는 ViT스타일로 일반화시킨 모델로, downsampling을 없애고 feature resolution을 고정한 모델을 뜻한다.
다만 해당 repo에서는 convnext_isotropic의 코드만 있을 뿐 학습이나 추론에는 사용하지 않는다.

## training 수행방법


ImageNet데이터셋을 다운로드 받고 아래와 같은 계층으로 데이터를 정리한다.
```
-imagenet-1k
--train
---1
----00000001.JPEG
----00000002.JPEG
---2
----00000003.JPEG
----00000004.JPEG
---3
----00000005.JPEG
----00000006.JPEG
--val
---1
----00000007.JPEG
---2
----00000008.JPEG
---3
----00000009.JPEG
```

validation 데이터의 경우 val 폴더 내에 이미지들을 모아두고 module1에 있는 imagenet_val_setup.sh 파일을 압축을 풀어 jpg파일이 존재하는 폴더에 넣은 후 imagenet_val_setup.sh를 실행한다. (정상작동하지 않을 시 윈도우에서 시도)

training을 수행하기 위해서 main.py 파일을 실행한다.  
해당 코드를 수행하게되면 224x224사이즈의 ConvNeXt-T로 IMAGENET 1k에 대해 학습한다.  
만일 다른 모델을 실행하고자 한다면 main2.py의 499번째 줄의 convnext_tiny를  
|모델 이름|parameter|
|---|---|
|ConvNeXt-T|convnext_tiny|
|ConvNeXt-S|convnext_small|
|ConvNeXt-B|convnext_base|
|ConvNeXt-L|convnext_large|

## pretrained model 다운로드
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

## 이미지에 대한 classification 수행방법

image 폴더에 classification 수행할 모든 이미지 파일을 저장

| model description | image size | model name | args |
|:---:|:---:|:---:|:---:|
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

---

# Revision

## Motivation
Image Classificaiton 네트워크는 이미지를 보고 어떤 물체인지 판별하는 역할을 한다. 이를 위해서 우리는 JPEG로 저장된 이미지를 사용하여 학습을 진행한다. 이때, JPEG는 손실 압축 코덱으로 원본 이미지의 정보중 인간이 알아차릴 수 없는 정보를 삭제하여 용량을 줄인다. 보통은 hihg-frequency 대역의 이미지 정보들이 손실되는 경향이 강하다. 그러나 실제로 JPEG 이미지를 여러 qualiyt로 변환하였을 때, 인간은 약간의 차이는 확인할 수 있지만 큰 차이는 확인할 수 없다. 이에 착안하여 이미지의 원본에서 크게 차이나지 않는 낮은 quality의 이미지로도 학습을 시키면 더 높은 성능을 얻을 수 있을 것이라 판단하였다.

실제로 원본과 quality를 50으로 줄인 이미지를 비교하였을 때, 큰 차이가 없는것을 확인할 수 있었고,
두 이미지를 모델에 넣게되면 출력되는 결과값이 달라지는 것을 확인할 수 있었다. 추가로 grad-cam으로 확인하였을 때, image classification과는 크게 상관없는 위치의 데이터들의 activation이 높아지는 것 또한 확인할 수 있었다.

또 다른 방법의 augmentation이라고 생각할 수 있겠다.

사실 channel pruning을 하려고 했는데 channel별 activation을 구하고 activation 높은거 없애고 하는데 시간이 많이 걸려서 시간상 문제로 진행하지 못하였다.

## Pseudo-Code
모델에 대한 Pseudo-Code는 동일하며, 학습 방법에서 차이가 발생하였다.

> 100 epoch까지 일반적인 imagenet-1k dataset으로 학습  
> 150 epoch까지 50%의 quality로 낮춘 jpeg 파일 dataset으로 학습

## train
module1의 image_quality_down.py 실행  
image_quality_down.py는 raw_im_path와 target_path가 존재한다.   
기본적으로 raw_im_path는 './path/to/imagenet-1k/train'  
target_path는 './path/to/imagenet-1k/train_50'이다.

학습은 먼저 기존 데이터셋을 이용하여 100epoch까지 학습을 진행하기 위해 main.py에서 498~503번 line을 아래와 같이 변경하고 main.py를 실행한다.
```python
    model_name = "convnext_tiny"
    batch_size = 184
    data_path = "./path/to/imagenet-1k"
    output_dir = "./path/to/save_results"
        
    args = parser.parse_args(["--model", model_name, "--drop_path", "0.1", "--batch_size", str(batch_size), "--lr", "4e-3", "--update_freq", "4", "--model_ema", "true", "--model_ema_eval", "true", "--data_path", data_path, "--output_dir", output_dir, '--epochs', str(100)])
```

이후 130epoch까지 학습을 진행하기 위해 module1폴더 내의 image_quality_dwon.py를 실행하고, path/to내에 생성된 imagenet-1k_50에 train폴더와 val폴더를 복사한 후, 아래와 같이 main.py의 503번 line을 변경한다.
```python
    model_name = "convnext_tiny"
    batch_size = 184
    data_path = "./path/to/imagenet-1k_50"
    output_dir = "./path/to/save_results"
        
    args = parser.parse_args(["--model", model_name, "--drop_path", "0.1", "--batch_size", str(batch_size), "--lr", "4e-3", "--update_freq", "4", "--model_ema", "true", "--model_ema_eval", "true", "--data_path", data_path, "--output_dir", output_dir, '--epochs', str(150)])
```
이후 main.py를 실행하여 학습을 진행한다.

## Results
100epoch까지 기존 IMAGENET-1k 데이터셋으로 학습을 진행하였고,  
130epoch까지 학습시 성능이 82.9% -> 83.1%로 0.2% 상승하는것을 확인할 수 있었다.