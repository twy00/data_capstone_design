# GANs를 사용한 X-Ray 이미지 질병 인식 성능 향상

* 2020-1 데이터분석캡스톤디자인 
* 유태원 2017103739 소프트웨어융합학과

## Overview

* __의료 이미지 데이터 분석 기술의 발전__: 
X-Ray, MRI, CT 등 의료 이미지 데이터는 환자의 질병을 미리 예측하고 예방하는 Computer-Aided Diagnosis(CAD) 시스템에서 매우 중요한 역할을 한다. 전통적으로는 이미지의 모양, 색깔 등의 조합을 활용 하여 분석을 했지만, 이는 복잡하고 복합적인 문제를 해결하기에는 부족한 측면이 있다. 딥러닝 기술이 발 전하면서 고차원적인 이미지 분석을 통해 질병을 판단하는 것이 가능해졌고, 이는 앞으로 여러 의료 분야 및 타 분야에서도 깊이 활용될 것으로 기대된다.

* __Data Imbalance 문제__: 
딥러닝 모델을 활용하여 이미지의 클래스를 분류할 때, 데이터셋이 클래스별로 다양하고 명확해야 하는 등 의 조건들과 더불어 클래스별로 데이터의 개수가 균형을 맞추는 것 또한 중요하다. 데이터 수가 클래스마다 크게 차이가 나면, 딥러닝 모델의 학습 결과가 균형잡힌 데이터의 학습 결과에 크게 못미친다. 예를 들어, 오픈 데이터셋인 Fashion MNIST 데이터셋의 10개 클래스 중 2개의 데이터 수를 60,000개, 나머지를 20,000 개로 설정하여 LeNet-5 모델을 학습했을 때, 평균 정확도가 77.13%였다. 이는 모든 60,000개의 데이터셋을 활용했을 때의 정확도인 89.94%보다 많이 부족하다는 것을 알 수 있다. 따라서 이 문제를 해결하기 위해 부좃한 데이터의 수를 증폭시키는 등의 방법을 사용할 수 있다.



# 연구 진행 방법

* 사용되는 데이터셋: [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/), 이미지 크기: 160x160x3
* 데이터셋 예시
<img width="300" alt="chexpert" src="https://user-images.githubusercontent.com/32740643/85944991-a0e69680-b975-11ea-8816-68abcf0ab02f.png">
* 사용되는 클래스와 데이터 수(train + validation)는 다음과 같다
  * Atelectasis (5,400 + 400)
  * Cardiomegaly (5,400 + 400)
  * Consolidation (2,700 + 400)
  * Edema (9500 + 400)
  * Fleural Effusion (13,500 + 400)
  * No Finding (3,000 + 400)

## 모델 학습 프레임워크
  * 모델 학습 프레임워크는 아래 그림과 같은 구조로 이루어져있다.
<img width="600" alt="framework" src="https://user-images.githubusercontent.com/32740643/85945069-34b86280-b976-11ea-96c3-06c6afd95851.png">

  * __Step 1 Train Feature Extractor__: Imbalance 상황에서 이미지를 분류하는 CNN 모델을 학습한다. 이 과정에서 학습한 모델의 convolutional layer를 거쳐서 나오는 feature는 다음 단계에서 GAN 모델의 input으로 사용된다.

### Extractor 구조
  
|#|    Type    |   Patch Size   |   Output Size   | Activation Function | Batch Normalization|
|:-:|:------------:|:----------------:|:-----------------:|:----------------:|:------------------:|
|1|Input| | 160x160x3| | |
|2|Convolution| 3x3x64 |160x160x64| Leaky ReLu| O |
|3|Convolution| 3x3x64 |160x160x64| Leaky ReLu | O |
|4|Max Pool| 2x2 |80x80x64| | |
|5|Convolution| 3x3x128 |80x80x128| Leaky ReLu| O |
|6|Convolution| 3x3x128 |80x80x128| Leaky ReLu| O |
|7|Max Pool| 2x2 |40x40x128| | |
|8|Convolution| 3x3x128 |40x40x128| Leaky ReLu| O |
|9|Convolution| 3x3x128 |40x40x128| Leaky ReLu| O |
|10|Convolution| 3x3x256 |40x40x256| Leaky ReLu| O |
|11|Max Pool| 2x2 |20x20x256| | |
|12|Convolution| 3x3x256 |20x20x256| Leaky ReLu| O |
|13|Convolution| 3x3x256 |20x20x256| Leaky ReLu| O |
|14|Convolution| 3x3x512 |20x20x512| Leaky ReLu| O |
|__15__|__Max Pool__| __2x2__ |__10x10x512__| | |
|16|Convolution| 3x3x512 |10x10x512| Leaky ReLu| O |
|17|Convolution| 3x3x512 |10x10x512| Leaky ReLu| O |
|18|Convolution| 3x3x512 |10x10x512| Leaky ReLu| O |
|19|Max Pool| 2x2 |5x5x512| | |
|20|Flatten| | 1x1x12,800|||
|21|Fully Connected|4,096|1X4,096|Leaky ReLu|O|
|22|Fully Connected|512|1X512|Leaky ReLu|O|
|23|Fully Connected|6|1X6|Softmax|O|

* Extractor 구조의 15번 째 줄의 10x10x512 크기의 Feature를 추출하여 GANs 모델의 Input으로 사용한다.

* __Step 2 Train Feature Generator__: 1단계에서 학습한 CNN 모델로부터 이미지의 feature를 가져온다. 이 feature를 input으로 하여 GAN 모델을 학습한다. GAN 모델에서 사용되는 네트워크와 그 역할은 아래와 같다.
     * __Generator__: 실제 이미지의 feature와 비슷한 가짜 데이터를 생성한다.
     * __Discriminator__: Generator가 생성한 feature가 진짜인지 가짜인지 구분하도록 학습한다.
     * __Classifier__: Generator가 생성한 feature의 클래스를 잘 분류하도록 학습한다. Classifier의 구조는 Extractor의 16번 줄부터 23번 줄까지의 구조를 사용하며, Step 1에서 학습한 Weight 값을 불러와 사용한다.
 
### Generator 구조
|#|    Type    |   Patch Size   |   Output Size   | Activation Function | Batch Normalization|
|:-:|:------------:|:----------------:|:-----------------:|:----------------:|:------------------:|
|1|Input| | 1x100| | |
|2|Flatten| | 1x1x2,048|||
|3|Deconvolution| 5x5 |1x1x2,048| Leaky ReLu| O |
|4|Deconvolution| 5x5 |5x5x1,024| Leaky ReLu| O |
|5|Deconvolution| 5x5 |10x10x512| Leaky ReLu| O |
|6|Deconvolution| 5x5 |10x10x512| Tanh| O |

### Discriminator 구조
|#|    Type    |   Patch Size   |   Output Size   | Activation Function | Batch Normalization|
|:-:|:------------:|:----------------:|:-----------------:|:----------------:|:------------------:|
|1|Input| | 10x10x512| | |
|2|Convolution| 5x5x512 |10x10x512| Leaky ReLu| O |
|3|Convolution| 5x5x1,024 |5x5x1024| Leaky ReLu| O |
|4|Convolution| 5x5x2,048 |2x2x2,048| Leaky ReLu| O |
|5|Flatten| | 1x1x8,192|||
|6|Fully Connected|4,096|1x1,024|Leaky ReLu|O|
|7|Fully Connected|64|1x64|Leaky ReLu|O|
|8|Fully Connected|1|1x1|Sigmoid|O|

### Classifier 구조
|#|    Type    |   Patch Size   |   Output Size   | Activation Function | Batch Normalization|
|:-:|:------------:|:----------------:|:-----------------:|:----------------:|:------------------:|
|1|Input| | 10x10x512| | |
|2|Convolution| 3x3x512 |10x10x512| Leaky ReLu| O |
|3|Convolution| 3x3x512 |10x10x512| Leaky ReLu| O |
|4|Convolution| 3x3x512 |10x10x512| Leaky ReLu| O |
|5|Avg Pool| 2x2 |5x5x512| | |
|6|Flatten| | 1x1x12,800|||
|7|Fully Connected|4,096|1X4,096|Leaky ReLu|O|
|8|Fully Connected|512|1X512|Leaky ReLu|O|
|9|Fully Connected|6|1X6|Softmax|O|


# 결과

## Step 1: Feature Extractor학습
학습에 사용한 최적의 parameter와 분류 정확도는 다음과 같다. 이 때 분류 정확도는 이후 연구 결과와 비교하기 위한 Baseline으로 사용한다.
|    Learning Rate    |   Batch Size   |   Epoch   | Accuracy|
|:------------:|:----------------:|:-----------------:|:-------:|
|0.0001|40|50|__56.8%__|

## Step 2: Feature Generator 학습
학습에 사용한 최적의 parameter와 분류 정확도는 다음과 같다.
|    Generator Learning Rate    |Discriminator learning Rate |Classifier Learning Rate|   Step 당 Batch Size   | Accuracy|
|:------------:|:----------------:|:-----------------:|:-------:|:------------:|
|0.001|0.0001|0.0000001|40|__61.7%__|

* 이 때 Classifier는 2 step 마다 가짜 데이터를, 5 step마다 진짜 데이터를 학습하여 Generator와 Classifier Weight 값을 조정하도록 하였다. 
* 학습결과, Classifier의 분류 정확도가 56.8% -> __61.7__%로 향상된 것을 확인할 수 있었다.

# 결론 및 제언
* 이번 연구에서는 의료 이미지 데이터 분류에서 발생할 수 있는 Data Imbalance 문제를 GANs 학습 프레임 워크를 도입함으로써 해결하는 것을 목표로 했다. Extractor, Generator, Discriminator, Classifier로 구성된 프 레임워크를 사용했으며, 기존 분류 정확도 56.8%에서 61.7%로 향상시킬 수 있었다. 이를 통해 GANs이 데 이터 분류 학습 분야에서 학습 과정에 도움을 줄 수 있다는 것을 알게 되었다.

* 이번 연구에서 부족했던 점은, Generator가 이미지를 생성할 때 10x10x512 Feature를 그대로 사용한 것이 다. CNN 모델이 깊어질수록 사용하는 Filter가 많아지고 추상적인 Feature를 뽑아내게 되는데, 160x160x3 이미지에 512개의 특징을 추출하면 아무래도 Filter가 각 클래스 데이터의 일반적인 부분이 아닌 특정 데이 터에만 나타날 수 있는 부분을 Feature로 뽑아낼 가능성이 있다. 즉, 모델이 매우 Over Fitting될 가능성이 있는 것이다. 따라서 향후 연구에서는 차원이 높지 않은 CNN모델을 Extractor로 사용하거나, 의미 있는 Feature만 추출할 수 있도록 Pruning하는 등의 방법을 통해 이 문제를 해결해볼 수 있을 것이다.


