### RNN - Recurrent Neural Network

#### 순서가 있는 데이터를 처리하는데 강점을 가진 신경망
#### 앞 단계에서 학습한 결과를 다음 단계의 학습에 이용하는 것, 따라서 학습 데이터를 단계별로 구분하여 입력해야 한다.
#### RNN의 기본 신경망은 긴 단계의 데이터를 학습할 때 맨 뒤에서는 맨 앞의 정보를 잘 기억하지 못하는 특성이 있음, 이를 보완하기 위해 다양한 구조가 만들어졌는데 LSTM과 GRU가 있다.

> ##### LSTM(Long Short-Term Memory) - 보통의 LSTM Unit은 cell, input gate, output gate 그리고 forget get로 이루어져 있다. cell은 값을 기억하는 역할을 한다. 각각의 세 gate는 인공 뉴런으로 생각할 수 있는데 머신 러닝에서 배운 NN처럼 weighted sum을 activation function을 통해 계산한다. 게이트는 값의 흐름을 조절해주는 역할을 한다고 생각하면 될것 같다. LSTM은 알 수없는 크기 및 기간의 시간차가 주어진 시계열을 분류, 처리 및 예측하는 데 적합하다. LSTM은 전통적인 RNN을 학습 할 때 기울기 소멸 또는 폴발의 문제를 해결하기 위해 개발되었다.

> ##### GRU(Gated Recurrent Units)