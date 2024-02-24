---
layout: post
title: "[오디오 신호 처리] 10. Band Energy Ratio, Spectral Centroid, Spectral Spread(Bandwidth)"
image: https://i.ibb.co/80vfyzS/thumbnail.png
date: 2024-01-09
tags: 
categories: audio-signal-processing
use_math: true
---

<br><br>

## Band Energy Ratio
<!-- excerpt-start -->
Band energy ratio는 두 주파수 대역의 에너지 비율을 나타내는 개념으로, 일반적으로 낮은 주파수의 에너지가 얼마나 우세한지를 측정하는 척도입니다. 다음과 같은 공식으로 계산됩니다.
<!-- excerpt-start -->
<br>
<center> $ BER_{t} = \frac{\sum^{F - 1}_{n=1} m_{t}(n)^2}{\sum^{N}_{n=F} m_{t}(n)^2} $ </center>
<br>

여기서 $m_{t}(n)$은 frequency bin $n$과 frame $t$에서의 magnitude를 나타내며 $N$은 전체 frequency bin의 개수입니다. $F$는 높은 주파수와 낮은 주파수 대역을 나누는 기준이 되는 frequency bin인데, 주로 2000 Hz의 값을 사용합니다.

이러한 주파수 대역의 에너지 비율은 오디오 신호의 특성을 파악하는 데 도움이 되며 음악 분석, 음성 처리, 음향 신호 처리 등 다양한 응용 분야에서 사용될 수 있습니다. 아래의 그림은 세 개의 다른 곡에 대한 2000 Hz 기준의 band energy ratio입니다. 음악의 장르에 따라 band energy ratio의 크기에 차이가 있는 것을 볼 수 있습니다.

<p align="center">
  <img src="https://i.ibb.co/wLCynnn/ber.png" alt="band energy ratio" border="0">
</p>

<br><br>

## Spectral Centroid

Spectral centroid는 오디오 신호의 주파수 분포에서 중심 위치를 나타내는 특징(feature)입니다. 이는 주파수 스펙트럼의 가중 평균으로, 신호의 에너지가 어떤 주파수 대역에 집중되어 있는지를 알려주고 소리의 brightness에 대한 척도가 됩니다. 수학적으로는 다음과 같이 정의됩니다.

<br>
<center> $ SC_{t} = \frac{\sum^{N}_{n=1} m_{t}(n) \cdot n}{\sum^{N}_{n=1} m_{t}(n)} $ </center>
<br>

다음 그림은 세 개의 다른 곡에 대한 spectral centroid를 비교한 것입니다.

<p align="center">
  <img src="https://i.ibb.co/gJTK0Dp/sc.png" alt="spectral centroid" border="0">
</p>

<br><br>

## Spectral Spread (Spectral Bandwidth)

Spectral spread 또는 spectral bandwidth는 소리가 얼마나 넓은 주파수 범위를 포함하고 있는지를 나타내는 지표입니다. Spectral centroid로부터 각 주파수 대역까지의 거리에 대한 가중 평균으로 정의되며 식으로는 다음과 같이 표현됩니다.

<br>
<center> $ BW_{t} = \frac{\sum^{N}_{n=1} \vert n - SC_{t} \vert \cdot m_{t}(n)}{\sum^{N}_{n=1} m_{t}(n)} $ </center>
<br>

소리의 에너지가 다양한 주파수 대역에 퍼져 있으면 spectral spread가 크고 특정 주파수 영역에 집중되어 있으면 spectral spread가 작습니다. 이것은 음악이나 음향 처리에서 중요한 특성 중 하나이며, 다양한 음원이나 소리의 스펙트럼 특성을 분석할 때 사용됩니다. 아래 그림은 세 개의 다른 곡에 대한 spectral spread를 시각화한 것입니다.

<p align="center">
  <img src="https://i.ibb.co/xMW45Pp/bw.png" alt="spectral spread" border="0">
</p>

<br><br>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "Frequency-Domain Audio Features"](https://youtu.be/3-bjAoAxQ9o?feature=shared)