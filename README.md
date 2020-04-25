# Data Imbalance 상황에서 GAN을 사용한 이미지 분류 정확도 향상

* 유태원 2017103739 소프트웨어융합학과

## Overview

* Data Imbalance 상황은 이미지 분류 CNN 모델을 학습하는 데 방해 요인이 될 수 있다. 특히 의료 이미지 데이터를 사용해서 질병을 예측하는 모델같은 경우 더 높은 분류 정확도가 필요하다.
* 이번 프로젝트에서는 GAN을 사용하여 가짜 흉부 x-ray 이미지 feature를 생성하고, 이를 통해  흉부 x-ray 이미지를 사용한 CNN 질병 분류 모델의 정확도를 향상시키는 것을 목표로 한다.

## Methods

* 사용되는 데이터셋: [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/), 이미지 크기: 160x160x3
* 사용되는 클래스와 데이터 수(train + validation)는 다음과 같다
  * Atelectasis (5,400 + 400)
  * Cardiomegaly (5,400 + 400)
  * Consolidation (2,700 + 400)
  * Edema (9500 + 400)
  * Fleural Effusion (13,500 + 400)
  * No Finding (3,000 + 400)

* 이 프로젝트는 총 3단계로 진행된다. 
  1. Imbalance 상황에서 이미지를 분류하는 CNN 모델 (VGGNET-16)을 학습한다. 이 과정에서 학습한 모델의 convolutional layer를 거쳐서 나오는 feature는 다음 단계에서 GAN 모델의 input으로 사용된다.
  
  2. 1단계에서 학습한 CNN 모델로부터 이미지의 feature를 가져온다. 이 feature를 input으로 하여 GAN 모델을 학습한다. GAN 모델에서 사용되는 네트워크와 그 역할은 아래와 같다.
    ** __Generator__: 실제 이미지의 feature와 비슷한 가짜 데이터를 생성한다.
    ** __Discriminator__: Generator가 생성한 feature가 진짜인지 가짜인지 구분하도록 학습한다.
    ** __Classifier__: Generator가 생성한 feature의 클래스를 잘 분류하도록 학습한다.



  3. 실제 이미지의 feature, GAN이 생성한 feature를 input으로 하여 최종 CNN 모델 (VGGNET-16)을 학습한다.

## Schedule

| Contents                         | March | April |  May  | June  |   Progress   |
|----------------------------------|-------|-------|-------|-------|--------------|
|  Step 1 CNN 모델 학습 및 수정        |   O   |   O   |       |       |              |
|  Step 2 GAN 모델 학습 및 수정        |       |   O   |   O   |       |              |
|  Step 3 최종 CNN 모델 학습 및 수정    |       |       |   O   |   O   |              |


