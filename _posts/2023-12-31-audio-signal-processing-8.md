---
layout: post
title: "[오디오 신호 처리] 8. Mel Spectrogram"
image: https://i.ibb.co/py3QyVk/thumbnail.png
date: 2023-12-29
tags: 
categories: audio-signal-processing
use_math: true
---

<br><br>

## Mel spectrogram의 필요성
<!-- excerpt-start -->
인간의 청각은 주파수를 선형적(linear)으로 인식하는 것이 아니라 로그 스케일(log scale)로 인식합니다. 예를 들어, C2 음은 약 65.4 Hz, C3는 약 130.8 Hz의 주파수를 가지므로 두 음의 주파수 차이는 약 65.4 Hz입니다. 그리고 C5의 주파수는 약 523.2 Hz, D5의 주파수는 약 587.3 Hz로 두 음의 주파수 차이는 약 64.1 Hz입니다. 즉, 인간은 100 Hz 근처의 영역에서는 약 65 Hz의 주파수 차이를 한 옥타브 차이로 인식하지만 500 Hz 근처의 영역에서는 한 음정 차이로 인식하는 것입니다.

이와 같은 인간 청각 시스템의 특성을 정확하게 반영하는 것이 음악 및 음성 신호를 분석하고 처리할 때 도움이 되는 경우가 많기 때문에 spectrogram을 그대로 사용하는 것이 아니라 mel scale로 변환하여 사용하는 것이 필요합니다.

<br><br>

## Mel scale

인간의 청각 특성을 반영하여 주파수 간의 간격을 더 자연스럽게 나타내기 위해 도입한 것이 mel scale입니다. Mel scale에서의 1000 mels는 1000 Hz와 같습니다. 이 기준점을 사용하여 mel scale에서 다른 주파수를 계산하게 되는데, 그 공식은 다음과 같습니다.

<br>
<center> $ m = 2595 \cdot \log(1 + f / 700)  $ </center>
<br>
<center> $ f = 700 \cdot (10^{m/2595} - 1)  $ </center>
<br>

여기서 $f$는 주파수(Hz)이고 $m$은 해당 주파수를 mel로 변환한 값입니다. 이러한 공식은 실험적으로 얻은 결과들을 토대로 설계되었습니다. 아래 그림은 mel scale과 Hertz scale의 관계를 나타낸 그래프입니다.

<p align="center">
  <img src="https://i.ibb.co/vLL7YN0/mel-hz-plot.png" alt="mel-hz plot" border="0">
</p>

<br><br>

## Mel spectrogram을 생성하는 방법

Mel spectrogram을 생성하기 위해서는 먼저 mel band의 개수를 정하고, 그에 맞게 mel filter bank를 설계한 뒤 spectrogram에 적용해야 합니다.

### Mel band

Mel band는 mel scale에서 나오는 여러 주파수 대역을 나타냅니다. 예를 들어 mel band의 개수가 10개라는 것은 오디오 신호의 전체 주파수 범위를 10개의 대역으로 나눠서 mel spectrogram에 나타낸다는 의미입니다. Mel band의 개수는 mel spectrogram을 생성할 때 중요한 하이퍼파라미터 중 하나인데, 분석하려는 오디오 데이터의 특성과 작업의 목적에 따라 달라질 수 있습니다.

적은 수의 mel band를 사용하면 주파수 간의 간격이 크게 되어 세부적인 주파수 정보를 잃을 수 있습니다. 반면 많은 수의 mel band를 사용하면 상세한 주파수를 포착할 수 있지만 계산량이 늘어나고 모델이 주파수의 노이즈나 특정한 패턴에 과적합될 가능성이 높아집니다. 일반적으로는 20에서 128 사이의 값을 사용하는 것이 흔하고, 실험과 검증을 통해 최적의 mel band 수를 결정하는 것이 좋습니다.

### Mel filter bank

Mel filter bank는 mel spectrogram을 생성하기 위해 주파수를 mel로 변환하고 필터링하는 데 사용되는 필터의 모음입니다. 각각의 mel filter는 특정 주파수 범위에 대해 가중치를 부여하고, 필터들의 모음이 전체 주파수 스펙트럼을 mel 주파수 영역으로 분해하는 데 기여합니다. 일반적으로 mel filter bank의 각 필터는 삼각형 필터(triangular filter)의 형태를 가지고 있습니다. Mel filter bank는 다음과 같은 과정을 따라 만들어집니다.

1. 오디오 신호의 최저/최고 주파수를 Hz에서 mel로 변환합니다.
<br>
<img src="https://i.ibb.co/kQ7jMhB/mel-filter-banks-1.png" alt="mel-filter-banks-1" border="0">

2. 최저/최고 주파수 사이에 정해진 mel band의 개수만큼 동일한 간격의 점을 생성합니다. (그림의 예시는 5개)
<br>
<img src="https://i.ibb.co/jrYNFF6/mel-filter-banks-2.png" alt="mel-filter-banks-2" border="0">

3. 생성된 점들을 다시 Hz로 변환합니다.
<br>
<img src="https://i.ibb.co/5LKmz6v/mel-filter-banks-3.png" alt="mel-filter-banks-3" border="0">

4. 변환된 주파수를 가장 가까운 frequency bin으로 근사합니다. 이 주파수가 각 mel band의 중심이 됩니다.
<br>
<img src="https://i.ibb.co/PWzC09L/mel-filter-banks-4.png" alt="mel-filter-banks-4" border="0">

5. 각각의 중심을 기준으로 triangular filter들을 생성합니다.
<br>
<img src="https://i.ibb.co/D815vcb/mel-filter-banks-5.png" alt="mel-filter-banks-5" border="0">

### Mel filter bank의 행렬 연산

Mel filter bank의 적용은 이산화된(discrete) 디지털 신호에 대해서 이루어지므로 실제로는 행렬을 만들어서 곱하는 방식으로 연산하게 됩니다. Short-time Fourier transform으로 얻어진 spectrogram $Y$는 (Number of frequency bins, Number of frames)의 shape을 갖습니다 (참고 포스트: [Short-Time Fourier Transform (STFT)](/2023/12/29/audio-signal-processing-7/)). 따라서 mel filter bank의 행렬 $M$은 (Number of mel bands, Number of frequency bins)의 shape을 갖도록 만들어서 행렬곱 $MY$를 하면 (Number of bands, Number of frames)의 shape을 갖는 mel spectrogram이 얻어집니다.

<br><br>

## Mel spectrogram 생성과 시각화

위에서 설명한 mel spectrogram에 대한 이론을 기반으로 실제 생성과 시각화를 하면 어떻게 되는지 살펴보겠습니다. 예시로 사용한 데이터는 피아노로 도레미파솔라시도 음을 차례로 연주한 오디오 신호입니다. 먼저 아래의 그림은 각각 y축을 linear 스케일로 표시한 spectrogram, y축을 log 스케일로 표시한 spectrogram, 그리고 mel-spectrogram입니다.

<p align="center">
  <img src="https://i.ibb.co/309gkRv/spectrograms.png" alt="spectrograms" border="0">
</p>

y축을 linear 스케일로 표시한 spectrogram은 실제로 fundamental frequency가 있는 낮은 주파수 영역이 잘 구분되지 않고, y축을 log 스케일로 표시한 spectrogram은 상대적으로 낮은 주파수 영역이 상세하게 보이긴 하지만 frequency bin의 간격은 동일하기 때문에 주파수가 높아질수록 많은 수의 frequency bin들이 몰려 있습니다. 반면 mel-spectrogram은 인간이 음정의 차이를 청각적으로 인식하는 것과 유사하게 frequency bin이 나눠져 있는 것을 볼 수 있습니다.

위에서 그린 mel-spectrogram은 mel band의 수를 80개로 설정하여 그린 것입니다. 이번에는 mel band의 개수에 따라 mel-spectrogram이 어떻게 달라지는지 보겠습니다. 아래의 그림은 각각 mel band의 수가 10개, 80개일 때의 mel-spectrogram입니다.

<p align="center">
  <img src="https://i.ibb.co/w7J0H3x/num-mel-bands.png" alt="number of mel bands" border="0">
</p>

Mel band의 수가 10개일 때의 그림을 보면 전체 주파수 영역이 정확하게 10개로 나눠져 있는 것을 볼 수 있습니다. 이 경우 해상도가 낮아 각 음들이 다르게 연주되는 것이 구분되지 않습니다. 이 mel spectrogram은 python의 librosa 라이브러리를 사용해서 그린 것인데 실제로 사용한 mel filter bank를 시각화해보면 아래 그림과 같은 모양을 하고 있습니다.

<p align="center">
  <img src="https://i.ibb.co/m06mMsS/librosa-filter-bank.png" alt="librosa mel filter bank" border="0">
</p>

Mel filter들의 중심 주파수에서의 weight가 1.0이 아닌 것을 볼 수 있습니다. 이는 필터의 크기가 정규화(normalization)되었기 때문이며 각 필터의 상대적인 에너지가 일관되도록 보장합니다. 필요에 따라 정규화를 수행하지 않는 경우 중심에서의 weight 값이 1.0이 될 수 있지만, 대부분의 경우 정규화를 사용하여 필터의 일관된 에너지 특성을 유지합니다.

<br><br>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "Mel Spectrograms Explained Easily"](https://youtu.be/9GHCiiDLHQ4?feature=shared)