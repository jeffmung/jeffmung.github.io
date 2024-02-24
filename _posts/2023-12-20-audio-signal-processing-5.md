---
layout: post
title: "[오디오 신호 처리] 5. Time Domain Audio Features - AE, RMSE, ZCR"
image: https://i.ibb.co/Drr8pwr/thumbnail.png
date: 2023-12-20
tags: 
categories: audio-signal-processing
use_math: true
---

<br><br>

## Amplitude Envelope (AE)

프레임 단위로 amplitude envelope를 구할 때에는 각각의 프레임 내에 있는 샘플들의 amplitude들 중 최대값을 취합니다. 따라서 프레임 t에서의 amplitude envelope 값은 다음의 식으로 나타납니다.

<br>
<center> $AE_{t} = \max_{k = t \cdot K}^{(t+1) \cdot K - 1} s(k)$ </center>
<br>

여기서 $K$는 프레임의 길이(사이즈)이고 $s(k)$는 $k$ 번째 샘플의 amplitude 값입니다. 모든 프레임들에 대해 amplitude envelope를 얻으면 그 신호의 loudness에 대한 개략적인 정보를 알 수 있습니다. 그런데 amplitude envelope는 그 프레임 내의 최대값으로 얻어지므로 outlier에 취약합니다.

Amplitude envelope는 음악에서 악기의 특성을 구분하거나 장르 분류 등에 사용될 수 있고, 음성 신호에서 발화의 시작과 끝을 탐지하는 작업 등에도 활용되는 특징(feature)입니다.

<p align="center">
  <img src="https://i.ibb.co/8rT7WrJ/ae.png" alt="amplitude envelope">
</p>

위의 그림에서 보라색 선이 amplitude envelope를 표시한 것입니다. 음악의 장르와 사용된 악기에 따라 차이가 나타나는 것을 볼 수 있습니다.

<br><br>

## Root-Mean-Square Energy (RMS)

오디오 신호의 root-mean-square energy는 시간에 대한 평균적인 파워가 어떻게 되는지를 알려주고 loudness에 대한 정보도 제공해주는 특징입니다. 아래의 식과 같이 프레임 내의 모든 샘플들에 대한 RMS 값을 계산하여 얻을 수 있습니다.

<br>
<center> $RMS_{t} = \sqrt{\frac{1}{K} \cdot \sum_{k = t \cdot K}^{(t + 1) \cdot K - 1}s(k)^{2}}$ </center>
<br>

RMSE는 amplitude envelope와는 다르게 모든 샘플들에 대한 평균으로 계산되므로 outlier에 덜 민감하다는 장점이 있습니다. 그림에서 분홍색 선으로 표시된 것이 RMSE입니다.

<p align="center">
  <img src="https://i.ibb.co/zrqG52p/rmse.png" alt="root-mean-square energy">
</p>

<br><br>

## Zero-Crossing Rate (ZCR)

Zero-crossing rate는 신호가 양수에서 음수로, 혹은 음수에서 양수로 바뀌면서 zero-crossing이 일어나는 빈도를 나타냅니다. 높은 ZCR은 고주파 성분이 많거나 빠르게 변화하는 신호라는 것을 의미합니다. ZCR을 계산하는 식은 다음과 같습니다.

<br>
<center> $ZCR_{t} = \frac{1}{2} \cdot \sum_{k = t \cdot K}^{(t + 1) \cdot K - 1} | sgn(s(k)) - sgn(s(k + 1)) |$ </center>
<br>

여기서 $sgn(\cdot)$은 부호를 나타내며 (+)이면 1, (-)이면 -1, 0이면 0의 값을 갖습니다. ZCR은 purcussive sound의 뾰족하고 짧은 단발성(transient) 변화를 감지하는 데에 유용하고 음악 장르 분류와 구조 분석 등에 활용될 수 있
습니다. 또한 음성 신호에서 음성이 있는(voiced) 부분과 무음(unvoiced) 부분을 구분하는 데에도 도움이 될 수 있습니다. 음성 신호는 주로 주기적이며 부드럽게 변화하여 ZCR이 상대적으로 낮은 반면 무음 신호는 비주기적이고 급격한 변화가 많아 ZCR이 높아질 수 있는 특성을 가지고 있습니다.

<p align="center">
  <img src="https://i.ibb.co/HT7bxwM/zcr.png" alt="zero-crossing rate">
</p>

<br><br>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "Understanding Time Domain Audio Features"](https://youtu.be/SRrQ_v-OOSg?feature=shared)
