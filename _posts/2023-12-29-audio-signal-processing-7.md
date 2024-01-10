---
layout: post
title: "[오디오 신호 처리] 7. Short-Time Fourier Transform (STFT)"
image: https://i.ibb.co/YL87bBG/thumbnail.png
date: 2023-12-29
tags: 
categories: Audio-Signal-Processing
use_math: true
---

<br><br>

## Short-Time Fourier Transform(STFT)의 필요성

시간 도메인의 신호에 discrete Fourier transform을 적용하면 전체 신호에 대한 주파수 특성을 알 수 있지만 여기에는 시간에 대한 정보가 전혀 포함되어 있지 않습니다. 음악, 음성 등 실제 신호는 시간에 따라 다양하게 변하기 때문에 주파수 성분의 시간적인 변화를 파악하는 것이 중요합니다. 이를 위해 수행하는 것이 short-time Fourier transform(STFT)입니다.

<br><br>

## Short-Time Fourier Transform(STFT)의 방법

Short-time Fourier transform은 시간 도메인에서 신호를 작은 부분 구간으로 나누고 각각에 대해 discrete Fourier transform(DFT)을 수행하는 것으로 이루어집니다. 즉, frame을 나누고 각 frame마다 windowing을 적용한 뒤 DFT를 적용해주기만 하면 됩니다. Frame과 windowing, hop size, overlap 등에 대한 자세한 설명은 [오디오 특징의 추출 (How to Extract Audio Features)](/2023/12/12/audio-signal-processing-4/) 포스팅에 있습니다. STFT를 식으로 표현하면 다음과 같습니다.

<br>
<center> $ S(m, k) = \sum_{n=0}^{N-1} x(n+mH) \cdot w(n) \cdot e^{-i 2 \pi (\frac{k}{N})n}  $ </center>
<br>

여기서 $ w(n) $은 window function이고, $m$은 시간 도메인에서 $m$번째 frame이라는 것을 의미하며 $H$는 hop length입니다. 일반적인 discrete Fourier transform과 다른 점은 DFT에서는 $N$이 전체 샘플의 개수를 의미하지만 STFT에서는 frame size, 즉 한 frame 내에 있는 샘플의 개수를 의미합니다. DFT에 대해서는 이전 포스트 [푸리에 변환 (Fourier Transform)](/2023/12/26/audio-signal-processing-6/)에 자세하게 설명되어 있습니다.

결과적으로 STFT를 수행하면 총 $m \times k$개의 Fourier coefficients가 얻어지며 각각은 그 시간과 주파수에 대한 magnitude와 phase 정보를 담고 있습니다. STFT의 파라미터 중 frame size의 설정은 결과에 큰 영향을 줄 수 있는데, frame size가 커지면 주파수에 대한 해상도(frequency resolution)는 커지지만 시간에 대한 해상도(time resolution)는 작아집니다. 반대로 frame size가 작아지면 frequency resolution은 작아지고 time resolution은 커지기 때문에 하려는 작업에 따라 적절한 frame size를 선택하는 것이 중요합니다.

<br><br>

## Short-Time Fourier Transform(STFT)의 결과 시각화

STFT의 큰 장점 중 하나는 소리를 효과적으로 시각화할 수 있다는 것입니다. STFT로 얻어진 Fourier coefficient의 크기(magnitude) 정보를 가지고 가로축이 시간, 세로축이 주파수인 형태로 heatmap을 그리면 spectrogram을 얻을 수 있습니다. 일반적으로 spectrogram의 amplitude는 로그 스케일, 즉 dB로 나타내며 주파수 또한 로그 스케일로 나타내는 경우가 많습니다.

<p align="center">
  <img src="https://i.ibb.co/xjcB3cJ/spectrogram.png" alt="spectrogram" border="0">
</p>

<br><br>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "Short-Time Fourier Transform Explained Easily"](https://youtu.be/-Yxj3yfvY-4?feature=shared)