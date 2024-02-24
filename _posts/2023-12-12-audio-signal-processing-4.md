---
layout: post
title: "[오디오 신호 처리] 4. 오디오 특징의 추출 (Audio Features Extraction)"
image: https://i.ibb.co/P58fq4K/thumbnail.png
date: 2023-12-12
tags: 
categories: audio-signal-processing
use_math: true
---

<br><br>

## Time Domain Feature Pipeline

오디오 신호를 다룰 때 time domain feature를 추출하는 기본적인 파이프라인을 살펴보겠습니다. 먼저 소리에 대한 아날로그 신호에 아날로그-디지털 변환(analogue to digital conversion, ADC)을 적용하여 디지털 신호를 얻습니다. 다음으로는 프레임(frame)을 만드는 작업이 필요합니다. 프레임은 연속적인 샘플들을 일정한 개수만큼 연결해놓은 작은 조각입니다.

### Frame

프레임 기반으로 오디오 신호를 처리하는 이유는 시간에 따라 변하는 오디오의 특성을 잘 포착할 수 있기 때문입니다. 예를 들어, 44.1 kHz의 sampling rate를 갖는 디지털 신호는 0.0227 ms의 샘플 길이를 갖습니다. 이는 사람이 청각적으로 인지할 수 있는 최소 길이인 10 ms보다 훨씬 작은 크기로, 음악이나 음성에서의 시간적인 변화를 포착하기에 너무 짧습니다. 만약 프레임 크기가 512라면 한 프레임의 길이는 약 11.6 ms가 됩니다. 이와 같이 너무 짧지 않은 적절한 시간의 프레임 기반으로 오디오 신호를 처리하면 의미 있는 시간적 변화를 잘 포착할 수 있습니다.

<p align="center">
  <img src="https://i.ibb.co/HBW8mkY/frame.png" alt="framing of audio signal processing">
</p>

일반적으로 프레임 크기는 2의 거듭제곱으로 설정하는데, 이것은 시간 도메인에서 주파수 도메인으로 변환하는 Fast Fourier Transform(FFT) 알고리즘과 관련이 있습니다. FFT 알고리즘의 특성상 입력 신호의 길이가 2의 거듭제곱일 때 특히 빠르게 동작하고 메모리 사용도 효율적입니다. 주로 sample rate에 따라 한 프레임의 길이가 10 ms에서 30 ms 정도가 되는 크기가 되도록 합니다.

또한 위 그림의 예시를 보면 프레임들이 서로 겹치게(overlap) 만들어져 있는데, 이렇게 하는 것이 주파수 도메인에서의 정확한 분석을 도와주기 때문입니다. 이 부분은 뒤에서 좀 더 자세하게 다루도록 하겠습니다.

이렇게 프레임 단위로 나누어 원하는 time domain feature를 연산하고 이러한 각 프레임들에 대한 feature를 집계하여(aggregate) 전체 오디오 신호에 대한 대표적인 feature를 얻습니다. 이 때 사용되는 집계 방법에는 평균, 중앙값, Gaussain Mixture Model(GMM) 등이 있습니다.

<br><br>

## Frequency Domain Feature Pipeline

Frequency domain에서 오디오 신호를 다루기 위해서는 time domain의 신호를 frequency domain으로 변환시켜야 합니다. 이를 위해 Fast Fourier Transform(FFT)를 주로 적용하게 되는데 세부적인 알고리즘에 대한 내용은 이후에 따로 다루도록 하고, 이때 발생할 수 있는 문제점들과 해결방법에 대해 먼저 알아보도록 하겠습니다.

### Spectral Leakage

FFT를 적용할 때에 고려해야 될 문제는 spectral leakage입니다. 일반적으로 우리가 다루는 유한한 길이의 오디오 신호를 생각해보면 신호의 길이가 정확하게 주기(period)의 정수배가 되는 일은 거의 없습니다. 따라서 양 끝 부분이 완전한 주기성을 갖지 못하고 불연속적으로 잘리게 되는데, 이 잘린 불연속적인 끝 구간은 주파수 관점에서보면 마치 높은 주파수를 갖는 것처럼 나타납니다. 이러한 고주파수 성분은 당연히 원래 신호에는 존재하지 않는 왜곡이기 때문에 frequency domain에서의 정확한 분석을 방해합니다.

### Windowing

이러한 spectral leakage를 감소시키기 위해 사용하는 방법 중 하나는 windowing입니다. Window function은 일종의 가중치를 부여하는 함수로 적용하였을 때 신호 프레임의 양 끝이 부드럽게 감소하도록 설계되어 있습니다. 일반적으로 사용하는 window function에는 Hann window, Hamming window, Blackman window 등이 있습니다. 그 중 Hann window는 다음과 같은 형태를 갖습니다.

<br>
<center> $w(k) = 0.5 \cdot (1 - \cos(\frac{2\pi k}{K - 1})), \quad k = 1, \cdots, K$ </center>

<p align="center">
  <img src="https://i.ibb.co/XVf8hGP/hann-window.png" alt="hann window">
</p>

w(k)는 신호에 곱해주는 weight이며 k는 프레임 안에 있는 샘플 변수입니다. 양 끝이 감소하는 종 모양이기 때문에 이러한 window function을 적용하면 신호의 양 끝이 0에 가깝게 줄어들고 이 부분에서 spectral leakage를 효과적으로 방지할 수 있습니다.

<p align="center">
  <img src="https://i.ibb.co/y401Pf2/windowing.png" alt="windowing">
</p>

### Overlapping Frames

Windowing을 적용하여 spectral leakage의 문제는 해결할 수 있지만 여기서 또다른 문제가 발생합니다. 프레임의 양 끝이 window function으로 인해 0에 가깝게 줄어든다는 것의 의미는 그 부분의 정보가 모두 손실된다는 것입니다. 따라서 이러한 정보의 손실을 막으려면 프레임들을 overlapping 하는 것이 필요합니다. 이렇게 프레임을 overlapping할 때 새로운 프레임이 시작되는 위치와 이전 프레임의 끝 위치 사이의 간격을 hop length라고 합니다. Hop length는 이웃하는 프레임 간의 중첩 정도를 결정하게 되고 일반적으로 샘플 단위로 표현됩니다.

<p align="center">
  <img src="https://i.ibb.co/bgSbFjG/hop-length.png" alt="hop length and frame size">
</p>

<br><br>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "How to Extract Audio Features"](https://youtu.be/8A-W1xk7qs8?feature=shared)