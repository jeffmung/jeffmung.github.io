---
layout: post
title: "[오디오 신호 처리] 2. 오디오 신호의 이해 - 아날로그/디지털 변환"
image: https://i.ibb.co/mzGjQ6n/audio-signal.png
date: 2023-12-08
tags: 
categories: Audio-Signal-Processing
use_math: true
---

<br><br>

## 오디오 신호

소리를 녹음하고 재생하려면 소리에 들어있는 모든 정보를 인코딩하고 다시 그 정보를 이용하여 같은 소리를 재현해내는 것이 필요합니다. 이러한 소리의 정보를 담고있는 표현이 오디오 신호이며 일반적으로 아날로그 신호의 경우에는 전압의 변화로, 디지털 신호의 경우에는 이진수의 배열로 나타냅니다. 아날로그 신호는 연속적인(continuous) 시간에 대해서 연속적인 실수값의 진폭을 갖습니다. 따라서 정보의 양이 무한하게 되고, 이것을 어떤 메모리에 저장하고 처리하는 것이 불가능하기 때문에 디지털 신호로 변환하는 것이 필요합니다. 디지털 신호는 이산적인(discrete) 시간에 대해서 유한한 개수의 값을 갖게 됩니다.

<p align="center">
  <img src="https://i.ibb.co/7K8pBTj/analogue-digital-signal.png" alt="analogue and digital signals>
</p>

<br><br>

## Analogue to Digital Conversion (ADC)

아날로그에서 디지털 신호로 변환하기 위해서는 sampling과 quantization을 해야 합니다.

### Sampling

Sampling은 연속적인 시간의 파동으로부터 일정한 시간 간격으로 값을 가져오는 것입니다. Sampling의 기준을 표현할 때에는 일반적으로 sampling rate라는 개념을 사용하는데, sampling 주기가 $T$라면 sampling rate는 $1/T$, 즉 Hz의 단위로 측정되는 값이 됩니다. Sampling rate가 높을수록 정보의 손실이 적어집니다.

<p align="center">
  <img src="https://i.ibb.co/SBnfpMs/samplingrate.png" alt="sampling rate">
</p>

그렇다면 어떠한 기준에 의해 sampling rate를 정해야 되는 것일까요? 예를 들어, CD의 sampling rate는 44,100 Hz입니다. 이것을 이해하기 위해서는 aliasing을 알아야 합니다. Aliasing은 sampling으로 인해 원래의 신호를 복원하지 못하고 왜곡되는 현상입니다. 아래 그림처럼 sampling rate가 낮을 경우 실제 신호의 주파수보다 더 낮은 주파수로 보이게 됩니다. 

<p align="center">
  <img src="https://i.ibb.co/dgt2VNq/aliasing.png" alt="aliasing">
</p>

이러한 aliasing 현상이 생기지 않게 하는 주파수의 경계를 Nyquist frequency라고 하며 그 크기는 sampling rate의 1/2배와 같습니다. 따라서 CD의 Nyquist frequency는 22,050 Hz가 되고, 22,050 Hz보다 낮은 주파수에서는 aliasing 현상 없이 원래 신호를 그대로 복원할 수 있습니다. 그리고 이 수치는 인간이 들을 수 있는 가장 높은 주파수에 해당하기 때문에 44,100 Hz의 sampling rate가 적절한 값이 되는 것입니다.

### Quantization

디지털 신호를 만들기 위해서는 시간에 대해서 값을 가져오는 sampling과 마찬가지로 진폭에 대해서도 무한한 실수 범위 내에서가 아닌 유한한 개수의 이산적인 값을 가져와야 합니다. 이것을 quantization이라고 하며 이진법의 bit로 정보를 처리하도록 만듭니다. 예를 들어, CD는 16 bit의 bit depth를 가지며 이는 아날로그 신호에서 sampling한 각각의 값들을 실수 그대로가 아닌 정해진 $2^{16}=65536$개의 숫자들 중에 가장 가까운 것으로 선택해서 기록한다는 의미입니다.

<p align="center">
  <img src="https://i.ibb.co/b5MGv6d/quantization.png" alt="quantization">
</p>

### Signal-to-Quantization Noise Ratio (SQNR)

아날로그 신호가 quantization 과정을 통해 디지털로 표현되면 원래의 값과 차이가 발생하게 되고 이를 quantization noise라고 합니다. 그리고 SQNR은 원래의 아날로그 신호와 quantization noise 사이의 비율을 나타냅니다. 높은 SQNR은 높은 정밀도를 나타내며 적은 quantization noise가 있다는 것을 의미합니다.

### Dynamic range

Dynamic range는 디지털 신호에서 처리할 수 있는 최대 신호 강도와 최소 신호 강도의 차이를 나타내며 dB로 표현합니다. Dynamic range가 큰 경우 시스템은 큰 신호와 작은 신호를 구별하기 쉽습니다. 높은 SQNR은 작은 신호도 정확하게 나타낼 수 있다는 것을 의미하고, 따라서 더 큰 dynamic range를 확보할 수 있습니다. 이는 음악 녹음이나 음향 시스템에서 중요하며 높은 dynamic range는 세밀한 음악 표현이 가능하도록 합니다.

<br><br>

## 소리의 녹음과 재생

이제 우리는 소리를 녹음해서 다시 재생하는 과정을 아날로그-디지털 변환(Analogue to Digital Conversion, ADC)과 디지털-아날로그 변환(Digital to Analogue Conversion, DAC)의 관점에서 이해할 수 있습니다.

### Analogue to Digital Conversion (ADC)

소리는 공기에서 발생한 압력의 변화로 표현되는 아날로그 신호입니다. 마이크는 이 압력 변화를 감지하고 해당 신호를 아날로그 전압 신호로 변환합니다.

오디오 인터페이스의 analogue to digital converter에서 아날로그 신호를 시간에 대해 일정한 간격으로 sampling합니다. 일반적으로 사람의 청각 주파수 범위에 맞게 최소 44.1 kHz의 샘플링 속도가 사용됩니다.

연속적인 아날로그 값들은 특정 수준의 이산적인 디지털 값으로 변환하는 quantization 과정을 거친 뒤 이진수로 표현되어 부호화(encoding) 됩니다. 컴퓨터가 처리할 수 있는 형태의 디지털 음향 데이터가 만들어진 것입니다.

### Digital to Analogue Conversion (DAC)

디지털 신호가 아날로그 신호로 다시 변환되어 연속적인 전압 변화를 나타냅니다. 변환된 아날로그 신호는 필요에 따라 증폭기(amplifier)를 통과하여 적절한 강도로 조절되고 필터링이 적용되어 불필요한 주파수 성분을 제거합니다. 최종적으로 변환된 아날로그 신호는 스피커를 통해 공기의 압력을 변화시켜 소리로 재생됩니다.

<p align="center">
  <img src="https://i.ibb.co/LY5Y4Bt/adc-dac.png" alt="adc and dac">
</p>

<br><br>

## Reference

[[Youtube] Valerio Velardo - The Sound of AI, "Understanding Audio Signals for Machine Learning"](https://youtu.be/daB9naGBVv4?feature=shared)