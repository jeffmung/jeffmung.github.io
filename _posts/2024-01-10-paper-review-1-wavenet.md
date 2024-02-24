---
layout: post
title: "[논문 리뷰] WaveNet: A Generative Model for Raw Audio"
image: https://i.ibb.co/PFc9Pnw/thumbnail.png
date: 2024-01-10
tags: 
categories: paper-review
use_math: true
---

<br><br>

## 논문 개요
<!-- excerpt-start -->
WaveNet은 2016년 구글 딥마인드에서 발표한 오디오 생성 모델입니다. 이 논문의 1저자인 Aaron van den Oord는 이전에 이미지를 생성하기 위한 모델인 PixelRNN과 ([van den Oord et. al., 2016](https://arxiv.org/abs/1601.06759)) PixelCNN을 ([van den Oord et. al., 2016](https://proceedings.neurips.cc/paper_files/paper/2016/hash/b1301141feffabac455e1f90a7de2054-Abstract.html)) 발표하였는데 PixelCNN에서 사용한 dilated causal CNN의 개념을 오디오에도 마찬가지로 적용하여 좋은 결과를 얻어냈습니다.

WaveNet은 text-to-speech(TTS)에 사용되어 텍스트를 음성으로 변환할 수도 있고, 발화자에 대해 조건을 두어 하나의 모델로 여러 목소리를 생성할 수도 있습니다. 또한 음악 데이터로 WaveNet을 학습시켜 음악을 생성하는 것도 가능합니다.

<br><br>

## Probabilistic and Autoregressive Generative Model

WaveNet은 입력으로 들어온 오디오 데이터 시퀀스의 다음에 올 값들을 예측하도록 학습하고, 생성할 때에는 이전 출력값이 다음 입력으로 들어가는 autoregressive 방식으로 동작하는 모델입니다. 다음 값을 예측한다는 것을 좀 더 자세히 말하면, 이전 타임스텝의 샘플들에 조건을 두어 다음 타임스텝의 파형(waveform) 확률 분포를 예측한다는 것입니다. 파형 $\mathbf{x} = \{ x_1, \ldots, x_T \}$의 결합 확률(joint probability)은 다음과 같은 조건부 확률의 곱으로 표현됩니다.

<br>
<center> $ p(\mathbf{x}) = \prod_{t=1}^T p(x_t \vert x_1, \ldots, x_{t-1}) $ </center>
<br>

이러한 조건부 확률분포는 dilated causal CNN으로 모델링되어 네트워크의 출력에 소프트맥스(softmax)를 취한 것이 다음 값 $x_t$에 대한 카테고리형 분포(categorical distribution)로 나옵니다. 예측된 확률과 실제 $x_t$의 로그 우도(log-likelihood)를 최대화하도록 하기 위해 손실 함수로는 크로스 엔트로피(cross entropy)가 사용됩니다.

<br><br>

## Dilated Causal Convolutions

PixelCNN에서와 마찬가지로 WaveNet에는 dilated causal convolution이 사용됩니다. 먼저 causal convolution은 미래가 아닌 과거의 정보만을 가지고 값을 예측하도록 하는 것입니다. 즉, 모델이 타임스텝 $t$의 $p(x_t \vert x_1, \ldots, x_{t-1})$에 대해 추정할 때 $x_{t+1}, x_{t+2}, \ldots, x_{T}$는 입력으로 들어가지 않아야 합니다.

<p align="center">
  <img src="https://i.ibb.co/sFLDMMN/causal-convolution.png" alt="causal-convolution" border="0">
</p>

Causal convolution의 단점이라고 한다면 수용영역(receptive field)을 증가시키기 위해 층(layer)의 개수나 필터(filter)의 크기가 커져야 한다는 것입니다. 예를 들어, 위의 그림에서 수용영역의 크기는 5밖에 되지 않습니다. 오디오 분야에서는 특히 샘플 하나의 길이가 매우 짧으므로 많은 타임스텝의 샘플들에 걸친 패턴을 학습해야 하는데, 계산 비용을 과도하게 증가시키지 않으면서 수용영역을 효과적으로 키우는 방법 중 하나가 바로 dilated convolution을 사용하는 것입니다.

Dilated convolution은 입력 값을 일정 스텝씩 뛰어넘어서 모델에 넣는 것으로, 필터를 일정 스텝 간격으로 키우고 사이의 값을 0으로 채우는 방법으로 적용할 수 있습니다. 아래의 그림을 보면 직관적으로 이해하기 쉽습니다. 첫 번째 그림은 일반적인 convolution, 즉 dilation이 1인 경우이고 두 번째 그림은 dilation이 2인 dilated convolution입니다. 파란색 영역이 입력이고 초록색 영역이 출력을 나타냅니다.

<br>
<p align="center">
  <img src="https://i.ibb.co/h82Cts7/no-dilation.gif" alt="no-dilation" border="0"><img src="https://i.ibb.co/xDcV1j5/dilation.gif" alt="dilation" border="0">
</p>
<br>

Dilated convolution을 여러 층 쌓으면 몇 개의 층만으로도 빠르게 수용영역을 증가시킬 수 있습니다. 예를 들어 아래의 그림은 dilation이 1, 2, 4, 8인 convolution 층을 차례로 쌓은 것입니다. 이 경우 수용영역의 크기는 최종적으로 16이 됩니다.

<p align="center">
  <img src="https://i.ibb.co/vDvyGkN/dilation-double.png" alt="dilation-double" border="0">
</p>

이 논문에서는 기본 하이퍼파라미터로 dilation을 1, 2, 4, ..., 512까지 쌓은 것을 하나의 블록(block)으로 간주하고 그 블록을 다시 여러 개 쌓는 방식으로 WaveNet 모델을 구현합니다. 즉, 각 층의 dilation은 맨 아래부터 순서대로 다음과 같은 값을 갖습니다.

<br>
<center> $ 1, 2, 4, \ldots, 512, 1, 2, 4, \ldots, 512, 1, 2, 4, \ldots, 512, \ldots $ </center>
<br>

<br><br>

## $\mu$-law Companding Transformation

오디오 샘플의 조건부 분포 $p(x_{t} \vert x_1, \ldots, x_{t-1})$을 모델에서 에측할 때 연속적인 실수값을 출력하도록 할 수도 있지만 WaveNet에서는 소프트맥스를 사용한 카테고리형 분포로 출력하도록 합니다. 오디오 신호는 보통 각 타임스텝에서의 샘플 값이 16 bit의 정수로 저장되어 있는데, 이것은 소프트맥스 층이 65,536개의 가능한 모든 정수값에 대한 확률을 출력해야 한다는 것을 의미합니다. 그렇게 되면 너무 많은 계산량이 필요하게 되므로 이 논문에서는 $\mu$-law companding transformation을 사용해서 256개의 값으로 범위를 줄여줍니다.

사람의 청각은 오디오 신호를 선형(linear)이 아닌 로그의 형태로 인식합니다. 즉, 진폭 값이 큰 부분에서는 넓은 범위를 비슷하다고 인식하고 값이 작은 부분에서는 조금만 달라져도 차이가 크다고 인식합니다. 따라서 동일한 간격으로 묶어서 범주의 개수를 줄이면 청각 특성과 맞지 않는 오류가 발생하게 됩니다. 예를 들어 1부터 100까지의 값이 있을 때 1~10, 11~20, ..., 91~100을 각각 하나의 범주로 하여 10개의 값으로 양자화 하는 것이 아니라 1~2, 2~4, 4~8, ..., 60~100과 같이 값이 커질수록 범위를 늘려가면서 묶어줘야 인간의 청각 특성을 올바르게 반영할 수 있습니다.

<p align="center">
  <img src="https://i.ibb.co/NCMfqDC/mu-law-function.png" alt="mu-law-function" border="0">
</p>


위의 그래프는 $\mu$-law 함수를 나타낸 것입니다. 절대값이 큰 부분에서는 넓은 범위의 값이 비슷한 값으로 변환되고, 절대값이 작은 부분에서는 좁은 범위의 값만 비슷한 값으로 변환되는 형태를 갖는 것을 볼 수 있습니다. 입력 $x$에 대해 $\mu$-law 인코딩된 $f(x)$는 다음과 같은 식으로 얻어집니다.

<br>
<center> $ f(x) = \text{sign}(x) \frac{\ln (1 + \mu \vert x_t \vert)}{\ln (1 + \mu)} $ </center>
<br>

이때 $-1 < x_t < 1 $이고 $\mu = 256$입니다. 이렇게 변환된 값을 256개의 값으로 양자화하여 각각을 하나의 범주 레이블로 사용하고, 학습된 모델을 이용하여 신호를 생성할 때에는 출력값을 위의 식을 반대로 전개하여 다시 디코딩합니다. 논문에서는 특히 음성 신호에 대해 이러한 비선형적인(non-linear) 양자화가 단순한 선형적 양자화 방식보다 훨씬 더 비슷하게 원래 신호를 복원했다고 합니다.

<br><br>

## Gated Activation Units

WaveNet에서는 PixelCNN과 마찬가지로 gated activation unit을 사용합니다. 게이트(Gate)는 값을 다음 층으로 얼마나 보낼지를 결정해주는 역할을 합니다. 일반적으로 recurrent network에서 긴 시간의 이전 정보들을 유지하기 위해서 많이 사용하는데 이 논문에서는 오디오 신호를 처리할 때 ReLU를 사용하는 것보다 이러한 gated activation unit을 사용하는 것이 훨씬 더 좋은 결과를 보여줬다고 합니다. 다음 식과 같이 게이트에 해당하는 필터를 따로 학습시킨 뒤 시그모이드 함수로 0~1 사이의 값을 만들고 원소별 곱셈(element-wise multiplication)을 하는 것으로 구현됩니다.

<br>
<center> $ \mathbf{z} = \tanh (W_{f, k} * \mathbf{x}) \odot \sigma (W_{g, k} * \mathbf{x}) $ </center>
<br>

여기서 $*$은 컨볼루션 오퍼레이터, $\odot$은 원소별 곱셈 오퍼레이터, $\sigma(\cdot)$는 시그모이드 함수, $k$는 신경망의 층 인덱스, $f$와 $g$는 각각 필터와 게이트를 나타냅니다.

<br><br>

## Residual and Skip Connections

WaveNet은 일반적으로 크고 깊게 쌓은 CNN에서 학습 성능을 향상시켜주는 residual과 skip connection도 사용합니다. 아래 그림과 같은 구조로 구현되는데 각각의 층에서 dilated convolution을 거치지 않은 residual을 마지막에 더해서 다음 층의 입력으로 넘겨주고, 각각의 층에서 gated activation unit까지 지난 결과들을 모아서 최종적으로 전부 합해주는 skip connection의 형태입니다.

<p align="center">
  <img src="https://i.ibb.co/TbVKJ9Y/residual-skip-connection.png" alt="residual skip connection" border="0">
</p>

여기서 $1 \times 1$ convolution은 채널의 수를 맞춰주거나 비선형성을 추가해주기 위해 사용하는데 그림에서는 하나로 표현되었지만 residual sum으로 전달되는 $1 \times 1$ convolution과 skip connection으로 전달되는 $1 \times 1$ convolution은 서로 다른 층입니다.

<br><br>

## Conditional WaveNets

텍스트를 음성으로 바꾸려면 모델을 학습할 때 음성 신호 외에도 각 시간에 맞는 텍스트 정보를 같이 입력해줘야 합니다. 또한 서로 다른 목소리를 구분해서 음성을 생성하기 위해 발화자마다 서로 다른 인덱스를 부여하여 음성 신호와 같이 입력으로 넣어줘서 모델을 학습시킬 수도 있습니다. 이러한 추가적인 입력 정보를 활용한 conditional WaveNet은 다음과 같은 식의 조건부 확률 분포를 모델링합니다.

<br>
<center> $ p(\mathbf{x} \vert \mathbf{h}) = \prod_{t=1}^T p(x_t \vert x_1, \ldots, x_{t-1}, \mathbf{h}) $ </center>
<br>

추가적인 입력 정보는 global conditioning과 local conditioning의 두 가지 형식으로 나타낼 수 있습니다. Global conditioning은 하나의 오디오 신호에서 모든 타임스텝에 대해 하나의 잠재(latent) 벡터 $\mathbf{h}$가 같이 입력으로 들어가는 것입니다. 발화자를 특정지어 임베딩하는 것이 이 경우에 해당합니다. Gated activation function의 식에 global conditioning을 적용하면 다음과 같이 됩니다.

<br>
<center> $ \mathbf{z} = \tanh (W_{f, k} * \mathbf{x} + V_{f, k}^T \vert \mathbf{h}) \odot \sigma (W_{g, k} * \mathbf{x} + V_{g, k}^T \mathbf{h}) $ </center>
<br>

이때 $V_{\cdot, k}$는 학습 가능한 선형 사영(linear projection), 즉 완전 연결층(fully connected layer)을 나타내고 $ V_{\cdot, k}^T \mathbf{h} $는 시간 차원에 대해 브로드캐스팅해준 것입니다.

Local conditioning은 시계열 $h_{t}$가 각각의 타임스텝에 맞게 입력으로 같이 들어가는 것입니다. TTS 모델에서 텍스트의 임베딩 벡터가 이 경우에 해당됩니다. 이러한 텍스트는 일반적으로 오디오 신호보다 낮은 샘플링 주파수를 갖기 때문에 transposed convolutional network를 이용해서 텍스트 임베딩 $\mathbf{h}$를 오디오 신호와 같은 길이를 갖는 새로운 벡터 $\mathbf{y} = f(\mathbf{h}) $로 업샘플링을 해줍니다. 이때 gated activation unit의 식은 다음과 같이 됩니다.

<br>
<center> $ \mathbf{z} = \tanh (W_{f, k} * \mathbf{x} + V_{f, k} * \mathbf{y}) \odot \sigma (W_{g, k} * \mathbf{x} + V_{g, k} * \mathbf{y}) $ </center>
<br>

여기서는 $V_{\cdot,k} * \mathbf{y}$가 $1 * 1$ convolution이 됩니다. Transposed convolutional network를 사용해서 업샘플링을 하는 대신 $V_{\cdot, k} \mathbf{h}$를 그냥 시간 차원으로 반복해서 사용할 수도 있지만 실험 결과 이 경우 성능이 약간 악화되었다고 합니다.

<br><br>

## Context Stack

기본적인 WaveNet의 구조는 위에서 모두 설명하였습니다. 추가적으로 논문에서는 수용영역을 증가시킬 수 있는 다른 방법으로 context stack을 활용하는 것을 제안합니다. 설명이 자세하게 나와 있지 않고 context stack이 적용된 공개된 코드도 없어 정확하게 이해한 것이 맞는지 확실하지 않은 부분입니다.

Context stack은 오디오 신호에서 일정 길이의 부분을 CNN과 같은 신경망에 통과시켜 정보가 추출된 벡터입니다. 이렇게 추출된 벡터를 WaveNet의 입력에 추가적인 조건으로 사용하면 이전 타임스텝에서의 장기 의존성이 쌓인 정보를 활용할 수 있습니다. WaveNet의 구조에서 블록을 더 쌓는 것과의 차이가 얼마나 있을지 모호한데 아직까지 이것과 관련하여 명쾌하게 설명되어 있는 자료를 찾지는 못했습니다.

<br><br>

## 실험

논문에서는 세 가지 태스크에 대한 실험 결과를 보여줍니다. 태스크는 각각 여러 발화자에 대해 학습한 음성 생성, TTS, 음악 생성입니다. 실험 결과로 생성된 예제들은 [구글 딥마인드 블로그](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/)에서 확인 할 수 있습니다.

### Multi-Speaker Speech Generation

여러 발화자에 조건을 둔 음성 생성은 텍스트 없이 발화자의 ID만 원핫 벡터로 만들어 입력으로 넣어주는 방식으로 실험을 진행했습니다. 텍스트를 음성으로 옮기는 것이 아니기 때문에 생성된 오디오가 어떤 내용을 담고 있을지 예상할 수가 없는데, 실제로 딥마인드 블로그에 들어가서 생성된 예시를 들어보면 사람이 말하는 것과 비슷하긴 하지만 실제로 존재하지 않는 단어들을 담은 소리가 납니다.

데이터셋에는 109명의 서로 다른 발화자의 음성들이 담겨 있는데 발화자로 조건을 부여한 하나의 WaveNet 모델로 각각의 특징들을 추출해서 학습할 수 있는 결과를 보여줍니다. 아마 텍스트 조건까지 같이 부여하면 학습 성능이 좀 더 떨어져서 실험을 그렇게 진행하지 않은 것으로 추측되는데, 딥마인드 블로그에서 하나의 영어 문장을 발화자의 목소리를 바꿔가며 생성한 음성 예시를 보여주고 있기는 합니다.

### Text-To-Speech

TTS 실험은 발화자의 ID 없이 텍스트 조건만 부여하여 진행했습니다. 비교군은 선행 연구들인 HMM-driven unit selection concatenative과 [(Gonzalvo et. al., 2016)](https://research.google/pubs/recent-advances-in-google-real-time-hmm-driven-unit-selection-synthesizer/) LSTM-RNN-based statistical parametric [(Zen et. al., 2016)](https://arxiv.org/abs/1606.06061) 음성 생성 모델입니다.

실험 결과를 보면 WaveNet (L)과 WaveNet (L+F)가 있는데 L은 언어적 특성(linguistic feature) 조건을 의미하고 F는 텍스트로부터 logarithmic fundamental frequency($\log F_0$)를 예측하는 외부 모델을 학습시켜 조건으로 넣어주는 것을 의미합니다. $\log F_0$는 생성된 음성의 강세와 억양을 자연스럽게 만드는 것을 도와주기 때문에 평가 결과에 큰 영향을 줍니다. 기본적으로 입력해주는 언어적 특성은 비교군 모델에도 동일하게 사용되었습니다.

<p align="center">
  <img src="https://i.ibb.co/jL9Pg3Q/wavenet-tts-result.png" alt="wavenet tts result" border="0">
</p>

위의 그래프에 사용된 선호도 점수는 같은 문장을 입력으로 넣어 생성된 서로 다른 모델들의 음성 샘플들을 실험 참가자들에게 블라인드로 듣게 하여 더 자연스럽게 들리는 것을 선택하도록 한 것입니다. WaveNet이 비교군 모델들보다 더 좋은 성능을 보여주고 영어와 만다린 중국어 모두 $\log F_0$를 넣어줬을 때 훨씬 더 좋은 결과를 나타냅니다.

<p align="center">
  <img src="https://i.ibb.co/jGfbdC9/wavenet-tts-mos.png" alt="wavenet tts mos" border="0">
</p>

위의 표는 mean opinion score(MOS) 테스트의 결과입니다. 이 테스트는 실험 참가자가 생성된 음성이 얼마나 자연스러운지 1: Bad, 2: Poor, 3: Fair, 4:Good, 5: Excellent의 다섯 개 점수로 나눠서 평가한 것입니다. 모델로 생성된 것이 아닌 실제 자연 음성을 8-bit-$\mu$-law나 16-bit linear PCM으로 축소시켜 양자화하고 다시 복원한 것과도 크게 차이가 나지 않는 결과도 볼 수 있습니다.

### Music

음악 생성 실험은 두 개의 데이터셋으로 진행했습니다. MagnaTagATuna [(Law & Von Ahn, 2009)](https://dl.acm.org/doi/abs/10.1145/1518701.1518881) 데이터셋은 총 200 시간, 각각 29초 길이의 음악 클립들로 구성되어 있고 길이로 장르, 악기, 템포, 볼륨, 분위기의 정보를 담고 있는 태그가 부여되어 있습니다. YouTube piano 데이터셋은 유튜브 영상에서 얻은 총 60 시간 길이의 솔로 피아노 음악들로 구성되어 있습니다. 악기가 하나이기 때문에 YouTube piano 데이터셋의 학습 난이도가 훨씬 쉽습니다.

음악 생성에서의 한계는 수용영역의 크기가 너무 작다는 것입니다. TTS나 음성 생성 쪽에서의 실험은 수용영역의 크기를 수백 ms 정도로 설정하여도 좋은 결과를 얻을 수 있는데 음악 생성은 수 초 정도로 길게 설정하더라도 몇 초마다 악기나 볼륨, 장르, 음질 등이 바뀌기 때문에 자연스러운 음악이라고 느껴지기가 어렵습니다. 그래도 딥마인드 블로그에서 YouTube piano 데이터셋 기준으로 생성된 음악 샘플들을 들어보면 꽤 자연스럽게 생성된 것들도 있습니다.

또한 MagnaTagATuna 데이터셋과 같이 태그 정보가 있으면 음성 생성에서와 마찬가지로 장르나 악기 등의 조건을 부여하여 학습하는 것이 가능합니다. 논문에서는 실험 결과 이러한 태그 정보의 조건부 학습이 꽤 잘 작동했다고 하지만 이렇게 생성된 샘플이 공개되어 있거나 평가 결과를 보여주고 있지는 않습니다.

<br><br>

## Reference

[Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016.](https://arxiv.org/abs/1609.03499)

[Pytorch implementation of WaveNet](https://github.com/vincentherrmann/pytorch-wavenet)
