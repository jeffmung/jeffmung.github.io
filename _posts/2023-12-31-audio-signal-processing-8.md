---
layout: post
title: "[오디오 신호 처리] 8. Mel Spectrogram"
image: https://drive.google.com/uc?export=view&id=1gtPe6tYNAy9jS5a2v_5eKLLgA__xIqhP
date: 2023-12-29
tags: 
categories: Audio-Signal-Processing
use_math: true
---

<br><br>

## Mel spectrogram의 필요성

인간의 청각은 주파수를 선형적(linear)으로 인식하는 것이 아니라 로그 스케일(log scale)로 인식합니다. 예를 들어, C2 음은 약 65.4 Hz, C3는 약 130.8 Hz의 주파수를 가지므로 두 음의 주파수 차이는 약 65.4 Hz입니다. 그리고 C5의 주파수는 약 523.2 Hz, D5의 주파수는 약 587.3 Hz로 두 음의 주파수 차이는 약 64.1 Hz입니다. 즉, 인간은 100 Hz 근처의 영역에서는 약 65 Hz의 주파수 차이를 한 옥타브 차이로 인식하지만 500 Hz 근처의 영역에서는 한 음정 차이로 인식하는 것입니다.

이와 같은 인간 청각 시스템의 특성을 정확하게 반영하는 것이 음악 및 음성 신호를 분석하고 처리할 때 도움이 되는 경우가 많기 때문에 spectrogram을 그대로 사용하는 것이 아니라 mel scale로 변환하여 사용하는 것이 필요합니다.

