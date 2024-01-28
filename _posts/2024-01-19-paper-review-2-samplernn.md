---
layout: post
title: "[논문 리뷰] SampleRNN: An Unconditional End-to-End Neural Audio Generation Model"
image: https://i.ibb.co/cw9zNGn/thumbnail.png
date: 2024-01-19
tags: 
categories: Paper-Review
use_math: true
---

<br><br>

## 논문 개요

RNN은 시계열 데이터를 학습시키고 시퀀스를 생성하기에 적합한 모델입니다. 하지만 오디오 도메인에서는 시간적 해상도(temporal resolution)와 스케일과 관련된 어려움들이 존재합니다. 예를 들어, 일반적으로 오디오 신호의 샘플 레이트가 적어도 16 kHz는 되는데 이러한 신호에서 학습해야 되는 패턴은 아주 작은 스케일로 연속적인 몇 개의 샘플들 사이에도 있을 수 있고, 수백 ~ 수천 ms에 걸쳐서 존재할 수도 있습니다.

게다가 RNN이 한 번에 하나의 샘플씩 생성할 경우 이러한 패턴에 대한 시간적 해상도와 너무 차이나는 스케일로 인해 좋은 성능을 기대하기가 어렵습니다. WaveNet이 [(van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499) RNN이 아닌 dilated CNN을 사용한 것도 아마 이러한 이유 때문일 것입니다.

이 논문에서는 계층적인 구조로 RNN 모델을 설계하여 이러한 어려움을 극복합니다. 실험 결과에서는 음성과 음악 데이터셋에 대해 학습시켜 WaveNet과 비교했을 때 더 좋은 성능을 보여줍니다. 몬트리올의 Yoshua Bengio 그룹이 연구를 진행하여 ICLR 2017에 발표하였습니다.

<br><br>

## Hierarchical RNN Model for Audio Waveforms

SampleRNN은 파형(waveform) 샘플들의 시퀀스를 이전 타임스텝의 샘플들에 대한 조건부 확률들의 곱으로 모델링합니다. 식으로 나타내면 다음과 같습니다.

<br>
<center> $ p(X) = \prod_{i=0}^{T-1} p(x_{i+1} \vert x_1, \ldots, x_{i}) $ </center>
<br>

이를 위해 사용하는 RNN은 다음 식과 같이 표현됩니다.

<br>
<center> $ h_t = \mathcal{H} (h_{t-1}, x_{i=t}) $ </center>
<center> $ p(x_{i+1} \vert x_1, \ldots, x_{i}) = Softmax (MLP(h_t)) $ </center>
<br>

여기서 $\mathcal{H}$는 GRU나 LSTM 등 RNN 계열 네트워크들의 메모리 셀이고 $h_{t}$는 타임스텝 $t$에서의 은닉 상태(hidden state)입니다. 오디오 신호를 처리할 때의 어려움을 극복하기 위해 SampleRNN은 각각 다른 시간적 해상도에서 동작하는 RNN 모듈들의 계층 구조를 사용합니다.

가장 낮은 모듈은 샘플 레벨에서 각각의 연속된 샘플들을 처리하고 높은 레벨의 모듈들은 더 긴 시간 스케일, 즉 더 낮은 시간적 해상도로 동작합니다. 각각의 모듈들은 바로 아래 레벨의 모듈에 조건을 부여하고, 가장 낮은 레벨의 모듈은 샘플 레벨의 예측을 출력합니다. 전체 모델은 엔드-투-엔드(end-to-end)로 역전파(backpropagation)를 통해 학습됩니다.

<br><br>

## Frame-level Modules

가장 아래 레벨을 제외한 모듈들은 모두 프레임 레벨로 동작합니다. 이때 프레임 $f^{(k)}$는 겹쳐지지 않은(non-overlapping) $FS^{(k)}$ 개의 샘플들로 구성되어 있습니다. $FS$는 Frame Size를 말하며 위첨자 $(k)$는 k번째 레벨의 모듈을 의미합니다.

각 모듈의 RNN은 이전 프레임까지의 입력을 반영한 은닉 상태 $h_{t-1}^{(k)}$과 현재의 입력 $inp_t^{(k)}$를 받아서 메모리를 업데이트합니다. 이때 $t$는 실제 타임스텝이 아니라 각 레벨의 시간적 해상도를 반영한 시간 간격입니다. 예를 들어, 3번째 레벨의 프레임 크기 $FS^{(3)}$이 64이면 $h_{t-1}^{(3)}$과 $h_{t}^{(3)}$ 사이에는 실제로 64개의 타임스텝 차이가 존재하는 것입니다.

입력 $inp$은 가장 위 레벨 $k=K$의 모듈일 때에는 단순히 현재 프레임이고, 중간 레벨 $1 < k < K$의 모듈에 대해서는 바로 위 레벨의 출력 벡터와 현재 프레임의 선형 결합(linear combination)이 됩니다. 이것을 식으로 나타내면 다음과 같습니다.

<br>
<center> $ inp_t = \left\{ \begin{aligned} &W_x f_t^{(k)} + c_t^{(k+1)} \qquad\; 1 < k < K \\ &f_t^{(k=K)} \qquad\qquad\qquad k = K \end{aligned} \right. $ </center>
<br>
<center> $ h_t = \mathcal{H}(h_{t-1}, inp_t) $ </center>
<br>

각 레벨의 모듈들이 서로 다른 시간적 해상도로 동작하기 때문에 위 모듈에서의 출력 벡터 $c$를 바로 아래 모듈의 입력 조건으로 넣어줄 때 업샘플링을 해줘야 됩니다. 위아래 모듈의 시간적 해상도 비율을 $r^{(k)}$라고 하면 위 모듈의 출력이 $r^{(k)}$ 개의 벡터로 확장되어야 하고, 이러한 업샘플링은 아래의 식과 같이 선형 프로젝션(linear projection)을 통해 이루어집니다. 

<br>
<center> $ c_{(t-1)*r + j}^{(k)} = W_j h_t \qquad\qquad 1 \leq j \leq r $ </center>
<br>

이러한 계층적 구조를 표현한 아래의 전체 모델 그림을 보면 이해하는 데 도움이 됩니다. 이 그림은 $K=3$이고 $FS^{(2)}$가 4, $FS^{(3)}$가 16일 때의 예시입니다.

<p align="center">
  <img src="https://i.ibb.co/vhcN3Rt/samplernn-model.png" alt="samplernn model" border="0">
</p>

2번째 레벨과 3번째 레벨의 시간적 해상도 비율 $r^{(3)}$가 4이므로 3번째 레벨의 모듈에서 출력된 벡터는 4개의 벡터로 업샘플링되어 2번째 레벨의 모듈로 들어갑니다. 2번째 모듈의 입력은 위에서 업샘플링된 벡터와 프레임 크기 $FS^{(2)} = 4$의 현재 프레임 입력이 선형 결합된 것입니다.

<br><br>

## Sample-level Module

가장 아래 레벨의 모듈은 한 번에 하나의 샘플 $x_{i+1}$에 대한 확률 분포를 출력합니다. 입력은 상위 레벨의 모듈들과 마찬가지로 프레임 단위로 넣을 수 있지만 이 경우에는 겹쳐진(overlapped) 프레임이 들어갑니다. 예를 들어 위의 그림에서는 첫 번째 레벨의 모듈에서 프레임 크기 $FS^{(1)}$가 4이고, $x_{i+28}, \ldots, x_{i+31}$이 먼저 입력으로 들어가면 다음 입력으로는 $x_{i+29}, \ldots, x_{i+32}$가 들어갑니다.

$FS^{(1)}$의 값이 일반적으로 작기 때문에 굳이 RNN을 사용하지 않고 간단하게 MLP를 사용합니다. 이때 양자화된 정수 값을 갖는 샘플 $x_i$는 임베딩 레이어를 통해 벡터 $e_i$로 임베딩 되고 바로 위의 모듈에서 업샘플링되어 출력된 벡터와 선형 결합되어 MLP에 입력으로 들어갑니다. 이를 식으로 나타내면 다음과 같습니다.

<br>
<center> $ f_{i-1}^{(1)} = flatten([e_{i - FS^{(1)}}, \ldots, e_{i-1}]) $ </center>
<center> $ f_{i}^{(1)} = flatten([e_{i - FS^{(1)} + 1}, \ldots, e_{i}]) $ </center>
<br>
<center> $ inp_i^{(1)} = W_x^{(1)}f_i^{(1)} + c_i^{(2)} $ </center>
<br>

최종 출력은 아래의 식과 같이 softmax를 적용하여 카테고리형 분포에 대한 음의 로그 우도(negative log-likelihood, NLL)를 최소화 하는 손실함수로 학습시킵니다.

<br>
<center> $ p(x_{i+1} \vert x_1, \ldots, x_i) = Softmax(MLP(inp_i^{(1)})) $ </center>
<br>

학습이 끝나고 생성할 때에는 한 번에 하나씩 샘플을 생성하고 자기회귀적(auto-regressive)으로 이전 스텝의 출력을 입력으로 넣어줍니다.

<br><br>

## Output Quantization

카테고리형 분포로 softmax를 통해 학습시키기 위해 오디오 신호를 너무 크지 않은 개수의 값으로 양자화하는 것이 필요합니다. SampleRNN 논문에서는 256개의 동일한 간격을 갖는 값으로 단순한 선형 양자화(linear quantization)를 적용하는데 각각을 하나의 값으로 사용하는 대신 위 섹션에서 잠깐 언급한 것과 같이 한번 임베딩 레이어를 거쳐서 벡터로 만들어주는 것이 성능 향상에 도움이 되었다고 합니다.

임베딩을 하지 않고 그냥 실수값을 사용한 경우에는 랜덤 노이즈와 같은 소리가 생성되었다고 하니 차이가 꽤 큰 것으로 보입니다. WaveNet에서 $\mu$-law companding을 통해 비선형적인 양자화를 해주는 것과 같은 역할을 임베딩 레이어의 파라미터가 학습되면서 비슷하게 해주는 것 같습니다.

<br><br>

## Truncated BPTT

SampleRNN에서는 WaveNet처럼 CNN을 사용하는 것이 아니라 RNN을 사용하므로 시퀀스가 길어짐에 따라 발생할 수 있는 그래디언트 소실(gradient vanishing)과 그래디언트 폭발(gradient explosion)의 문제에서 자유로울 수 없습니다. 오디오 신호의 특성 상 장기 의존성(long-term dependency)을 효과적으로 학습하려면 시퀀스의 길이가 어느정도 길어져야 하는데, 그에 따른 한계를 극복하기 위해 truncated backpropagation through time(BPTT)을 사용합니다.

RNN을 학습할 때에는 입력이 시간에 따라 순서대로 들어가기 때문에 맨 마지막 타임스텝에서부터 맨 처음 타임스텝까지 손실함수에 대한 그래디언트를 전파시켜서 네트워크의 파라미터를 업데이트합니다. 이를 BPTT라고 하는데, 입력 시퀀스의 길이가 길어질수록 역전파 시 그래디언트가 한없이 작아지거나 급격하게 커지기 때문에 업데이트가 제대로 이루어지지 못합니다.

이러한 문제를 완화시키기 위한 truncated BPTT는 각각의 시퀀스를 일정한 길이의 짧은 서브시퀀스로 나누고, 역전파를 할 때에는 서브시퀀스 내에서만 그래디언트가 전파되도록 하는 방법입니다. 이렇게 역전파는 일정 길이로 끊어줌으로써 그래디언트 소실이나 폭발을 방지하고, 순전파를 할 때에는 이전 서브시퀀스의 맨 마지막 은닉 상태와 가중치(weight)를 저장해두었다가 다음 서브시퀀스의 첫 은닉 상태로 넘겨줘서 시간 순서에 따라 업데이트가 연속적으로 진행될 수 있도록 해줍니다.

<br><br>

## 실험

논문의 실험은 비교대상으로 주로 WaveNet을 사용합니다. 비교에 사용한 WaveNet은 공식적으로 코드가 공개되어 있지 않으므로 저자들이 최대한 하이퍼파라미터들을 맞춰서 재구현한 것입니다.

데이터셋으로는 세 가지를 사용하는데, Blizzard는 [(Prahallad et al., 2013)](http://festvox.org/blizzard/bc2013/blizzard_2013_summary_indian.pdf) 한 명의 여성 목소리로 이루어진 20.5 시간짜리 음성 데이터셋입니다. Onomatopoeia는 숨소리나 기침소리, 비명소리 등 짧은 인간의 발성으로 이루어진 3.5 시간짜리 데이터셋입니다. 이 데이터셋은 총 길이가 상대적으로 짧은 대신 51명의 서로 다른 사람들이 녹음한 많은 종류의 소리들이 포함되어 있다는 점에서 난이도가 높습니다. 마지막으로 Music은  [인터넷 아카이브](https://archive.org/)에 공개되어 있는 총 32개의 베토벤 피아노 소나타들로 구성된 10 시간짜리 데이터셋입니다.

아래의 그림은 각각의 데이터셋과 생성된 샘플의 시각화된 예시입니다. 위의 세줄은 2초, 아래의 3줄은 100 ms의 파형입니다. 모든 데이터셋은 16 kHz 샘플 레이트와 16 bit depth를 사용하고 각각 일정 비율로 training/validation/test set을 나눠서 사용했습니다. 논문에는 모델의 하이퍼파라미터와 데이터 전처리 등에 대한 더 세부적인 실험 조건들이 작성되어 있습니다.

<p align="center">
  <img src="https://i.ibb.co/sJZLJTV/dataset.png" alt="dataset" border="0">
</p>

### Ablations

Ablation 실험으로 값을 양자화할 때 임베딩 레이어를 사용하는 것과 계층 구조 맨 아래의 샘플 레벨 모듈을 사용하는 것의 영향을 분석했습니다. 아래 표에서 Without Embedding은 256 개의 정수 값으로 양자화를 하고 벡터로 한번 더 임베딩을 해주지는 않았을 때의 결과입니다. Multi-Softmax는 맨 아래의 MLP 레이어를 사용하지 않고 두 번째 레벨 모듈의 출력 벡터 $c$에 바로 소프트맥스를 적용해서 $FS^{(1)}$ 개의 전체 프레임을 한 번에 출력하는 경우의 결과입니다. 즉, 위의 Frame-level Modules 섹션에 있는 전체 모델 그림의 Tier 1을 없애고 Tier 2까지만 사용한 것입니다.

<p align="center">
  <img src="https://i.ibb.co/Dt0Nff3/ablation.png" alt="ablation" border="0">
</p>

NLL은 Negative Log-Likelihood로 값이 낮을수록 더 좋은 성능을 나타냅니다. 샘플 레벨 모듈과 양자화된 값의 임베딩을 모두 사용했을 때 성능이 가장 좋은 것을 볼 수 있습니다.

### Overall Results

NLL 값을 기준으로 평가한 실험 결과는 아래 표에 나와 있습니다. 또한 SampleRNN으로 생성된 예시 샘플들은 [Soundcloud](https://soundcloud.com/samplernn/sets)에 공개되어 있습니다.

<p align="center">
  <img src="https://i.ibb.co/9yBmZbg/overall-results.png" alt="overall results" border="0">
</p>

SampleRNN이 가장 좋은 성능을 보여주는데 흥미로운 것은 Blizzard와 Music 데이터셋에서 기본적인 RNN과 WaveNet의 성능이 유사합니다. SampleRNN 저자들이 WaveNet을 재구현하면서 최적화가 잘 되지 않았기 때문인지 실제로 RNN의 성능을 WaveNet의 dilated CNN이 따라가지 못하는 것인지는 검증이 필요해 보입니다. ICLR에 이 논문이 투고될 때의 [OpenReview](https://openreview.net/forum?id=SkxKPDv5xl)에도 리뷰어들이 이 점을 지적했는데, 저자들의 답변은 WaveNet 논문에 정보가 모두 제공되어 있지 않아 그대로 구현하는 것은 불가능하며 비교에 사용한 RNN에도 Truncated BPTT가 적용되어 오디오 신호의 긴 시퀀스에 대해 효과적일 수 있다는 것이었습니다.

### Human Evaluation

NLL을 기준으로 한 정량적인 평가 외에 사람이 생성된 샘플을 듣고 선호도를 비교한 테스트도 진행했습니다. 온라인으로 공개되어있는 Web Audio Evaluation Tool을 [(Jillings et al., 2015)](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/12628/Moffat%20WEB%20AUDIO%20EVALUATION%20TOOL%202015%20Published.pdf?sequence=2) 사용했습니다.


Blizzard 데이터셋에 대한 실험은 텍스트 등에 조건을 두고 생성한 것이 아니기 때문에 생성된 소리는 사람의 음성처럼 들리긴 하지만 의미는 없는 웅얼거림에 가깝습니다. 이때 평가에 참여한 사람들은 서로 다른 모델에서 생성된 두 개의 샘플 중 더 자연스럽고 사람의 음성과 비슷하게 들리는 것을 선택합니다. 그 결과는 아래의 그림과 같습니다.

<p align="center">
  <img src="https://i.ibb.co/Ld1jp7H/blizzard-human-eval.png" alt="blizzard human evaluation" border="0">
</p>

아래의 그림은 Music 데이터셋에 대한 실험 결과입니다. 이 실험에서는 생성된 오디오 샘플이 음악 또는 랜덤 노이즈의 두 가지 경우로 나뉘었는데 이 중 노이즈로 생성된 것은 제외하고 음악처럼 들리는 샘플들만 평가에 사용했습니다.

<p align="center">
  <img src="https://i.ibb.co/rkB4SxW/music-human-eval.png" alt="music human evaluation" border="0">
</p>

### Quantifying Information Retention

마지막 실험은 SampleRNN이 메모리를 어느 정도의 길이까지 유효하게 학습할 수 있는지에 대한 것입니다. 구체적으로, 3-tier SampleRNN을 각각 한명씩의 남성과 여성이 읽은 오디오북 데이터셋에 대해 학습시킵니다. 이때 발화자의 ID와 같은 정보는 전혀 주어지지 않는데 학습된 모델로부터 생성된 샘플이 일관적으로 같은 발화자의 목소리를 유지하는지 확인하는 것입니다. 실험 결과 생성된 샘플은 항상 처음부터 끝까지 같은 목소리로 유지되었다고 합니다.

또한 샘플을 생성할 때 1초의 공백을 중간에 넣어주는 실험도 진행했습니다. 초반 2초 동안은 일반적인 자기회귀적인 방법으로 샘플을 생성하고, 2초부터 3초까지는 이전에 생성된 샘플을 다시 입력으로 넣어주는 대신에 0의 값을 넣어줍니다. 그리고 3초부터 5초까지는 다시 일반적인 방법으로 생성합니다. 이와 같은 실험의 목적은 모델의 메모리가 어느 정도의 길이(horizon)까지 유효한지를 확인하기 위한 것입니다.

생성된 샘플의 처음 2초와 마지막 2초 구간이 일치하는지를 기준으로 평가하였을 때 SampleRNN은 같은 목소리로 생성된 비율이 83%였습니다. WaveNet과 같이 정해진 길이의 수용영역(receptive field)를 사용하는 경우에는 그 길이보다 긴 공백이 있을 때 유효한 메모리의 역할을 하지 못합니다. 예를 들어 1초의 공백은 16000개의 타임스텝이기 때문에 그보다 짧은 수용영역의 WaveNet이라면 같은 목소리가 유지되는 비율이 50%입니다. RNN을 기반으로 한 SampelRNN이 이러한 메모리의 유효 길이에 대해 강점을 가지고 있다는 것을 보여주는 실험 결과입니다.

<br><br>

## SampleRNN-WaveNet Hybrid

SampleRNN의 계층적인 구조와 WaveNet의 dilated CNN은 각각 장단점을 가지고 있습니다. 따라서 SampleRNN과 WaveNet을 혼합한 형태의 모델은 더 좋은 성능을 나타낼 수 있을지 자연스럽게 궁금증이 생깁니다. 논문의 appendix에도 이에 대한 내용이 나와 있습니다. 두 가지의 모델을 실험하였는데 첫 번째는 SampleRNN의 계층적인 구조를 적용하면서 모든 모듈을 다 WaveNet으로 구현한 것입니다. 두 번째는 프레임 레벨 모듈은 RNN으로, 샘플 레벨 모듈은 WaveNet으로 구현한 것입니다.

결과적으로는 두 가지의 모델 모두 기대와 달리 만족할 만한 성능을 보여주지 못했다고 합니다. 이 때문에 SampleRNN 논문에서는 결국 WaveNet의 dilated CNN을 차용하지 않고 모든 레벨의 모듈을 다 RNN 기반으로 구현했겠지만 세부적인 디자인에 따라 좋은 성능을 나타낼 수 있는 가능성은 충분히 있는 접근으로 보입니다.

<br><br>

## Reference

[Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Sotelo, Aaron Courville, and Yoshua Bengio. SampleRNN: An Unconditional End-to-End Neural Audio Generation Model. In International Conference on Learning Representations, 2017.](https://openreview.net/forum?id=SkxKPDv5xl)

[Pytorch implementation of SampleRNN](https://github.com/deepsound-project/samplernn-pytorch)
