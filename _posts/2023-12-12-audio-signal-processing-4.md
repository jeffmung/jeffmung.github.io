---
layout: post
title: "[오디오 신호 처리] 4. 오디오 특징의 추출 (How to Extract Audio Features)"
image: https://drive.google.com/uc?export=view&id=11-2bXqA1N69xgCgqZ5U3ylGvTyIu0EK7
date: 2023-12-12
tags: 
categories: Audio-Signal-Processing
use_math: true
---

<br><br>

#### Time Domain Feature Pipeline

오디오 신호를 다룰 때 time domain feature를 추출하는 기본적인 파이프라인을 살펴보겠습니다. 먼저 소리에 대한 아날로그 신호에 아날로그-디지털 변환(analogue to digital conversion, ADC)을 적용하여 디지털 신호를 얻습니다. 다음으로는 프레임(frame)을 만드는 작업이 필요합니다. 프레임은 연속적인 샘플들을 일정한 개수만큼 연결해놓은 작은 조각입니다.

프레임 기반으로 오디오 신호를 처리하는 이유는 시간에 따라 변하는 오디오의 특성을 잘 포착할 수 있기 때문입니다. 예를 들어, 44.1 kHz의 sampling rate를 갖는 디지털 신호는 0.0227 ms의 샘플 길이를 갖습니다. 이는 사람이 청각적으로 인지할 수 있는 최소 길이인 10 ms보다 훨씬 작은 크기로, 음악이나 음성에서의 시간적인 변화를 포착하기에 너무 짧습니다. 만약 프레임 크기가 512라면 한 프레임의 길이는 약 11.6 ms가 됩니다. 이와 같이 너무 짧지 않은 적절한 시간의 프레임 기반으로 오디오 신호를 처리하면 의미 있는 시간적 변화를 잘 포착할 수 있습니다.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1L_H2F0gfTn9nl8aQueR5Blsl46M5iify" alt="img1">
</p>

일반적으로 프레임 크기는 2의 거듭제곱으로 설정하는데, 이것은 시간 도메인에서 주파수 도메인으로 변환하는 Fast Fourier Transform(FFT) 알고리즘과 관련이 있습니다. FFT 알고리즘의 특성상 입력 신호의 길이가 2의 거듭제곱일 때 특히 빠르게 동작하고 메모리 사용도 효율적입니다. 주로 sample rate에 따라 한 프레임의 길이가 10 ms에서 30 ms 정도가 되는 크기가 되도록 합니다.

또한 위 그림의 예시를 보면 프레임들이 서로 겹치게(overlap) 만들어져 있는데, 이렇게 하는 것이 주파수 도메인에서의 정확한 분석을 도와주기 때문입니다. 이 부분은 뒤에서 좀 더 자세하게 다루도록 하겠습니다.

이렇게 프레임 단위로 나누어 원하는 time domain feature를 연산하고 이러한 각 프레임들에 대한 feature를 집계하여(aggregate) 전체 오디오 신호에 대한 대표적인 feature를 얻습니다. 이 때 사용되는 집계 방법에는 평균, 중앙값, Gaussain Mixture Model(GMM) 등이 있습니다.

<br><br>

#### Frequency Domain Feature Pipeline