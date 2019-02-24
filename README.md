### RNN - Recurrent Neural Network

#### 순서가 있는 데이터를 처리하는데 강점을 가진 신경망
#### 앞 단계에서 학습한 결과를 다음 단계의 학습에 이용하는 것, 따라서 학습 데이터를 단계별로 구분하여 입력해야 한다.
#### RNN의 기본 신경망은 긴 단계의 데이터를 학습할 때 맨 뒤에서는 맨 앞의 정보를 잘 기억하지 못하는 특성이 있음, 이를 보완하기 위해 다양한 구조가 만들어졌는데 LSTM과 GRU가 있다.

> ### LSTM(Long Short-Term Memory) - 보통의 LSTM Unit은 cell, input gate, output gate 그리고 forget get로 이루어져 있다. cell은 값을 기억하는 역할을 한다. 각각의 세 gate는 인공 뉴런으로 생각할 수 있는데 머신 러닝에서 배운 NN처럼 weighted sum을 activation function을 통해 계산한다. 게이트는 값의 흐름을 조절해주는 역할을 한다고 생각하면 될것 같다. LSTM은 알 수없는 크기 및 기간의 시간차가 주어진 시계열을 분류, 처리 및 예측하는 데 적합하다. LSTM은 전통적인 RNN을 학습 할 때 기울기 소멸 또는 폴발의 문제를 해결하기 위해 개발되었다.
> ##### 은닉뉴런을 Ht라하고 비선형성 함수를 f라 할때 다음과 같은 식이 나온다.
> ##### Ht = f(Wt*It + Wt-1*Ht-1)
> ##### 이걸 입력값으로 미분하면 다음과 같은 식이 나오는데
> ##### dHt/dIt-k = W(rec)^k*W(in) (W(rec)은 순환되는 W값)
> ##### k가 커질수록 0에 가까워지고 이는 기울기 손실(Vanishing Gradient)를 유발한다.
> ##### 따라서 이를 해결하기 위해 개발된 것이 LSTM이다.
```python
initial_state = cell.zero_state(batch_size, tf.float32) #LSTM에서 특정 메모리 셀이 유지되어야 하는지에 대한 비트 텐서가 필요하다.
output, _states = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
```
> ##### 시간 단계마다 한 LSTM 유닛은 세 단계의 새로운 정보로 메모리 셀을 수정하는데
> ##### 1. 유닛은 얼마나 많은 양의 이전 메모리를 유지할지를 결정해야 한다. 이것은 유지 게이트(keep gate)에 의해 결정된다. 개념은, 이전 시간 단계에서 메모리 상태 텐서는 정보가 풍부하지만 일부 정보는 오래된 것일 수 있으므로 삭제되어야 한다. 이전 상태와 곱해지는 비트 텐서에 대한 계산을 시도한다. 필요가 없다면 비트텐서가 0이므로 0을 곱할 것이고 필요하다면 비트 텐서가 1이므로 1을 곱할 것이다.

> ##### 2. 어떤 정보를 메모리 상태에 기록할지 생각해야 한다. LSTM 유닛의 이 부분을 쓰기 게이트(wirte gate)라고 한다. 크게 두 부분으로 나뉘는데 첫번째 구성 요소는 어떤 정보를 쓰려고 하는지를 알아낸다. 이것은 중간 텐서를 생성하기 위해 tanh층에서 계산된다. 두번째 구성요소는 계산된 텐서가 새로운 상태를 포함하기를 원하는지 아니면 단발성으로 전달하길 원하는지를 파악한다.역시 비트 벡터를 근사해 이 작업을 수행한다.

> ##### 3. 시간 단계마다 LSTM 유닛이 출력을 제공하기를 바란다. 출력 게이트(output gate)의 구조는 쓰기 게이트와 거의 유사하다. 1. tanh 층은 상태 벡터로부터 하나의 중간 텐서를 생성, 2. 시그모이드층은 현재 입력과 이전 출력을 사용해 비트 텐서 마스크 생성, 3. 중간 텐서는 최종 출력을 생성하기 위해 비트 텐서와 곱해진다.

> ##### 이것이 그냥 RNN을 쓰는 것보다 나은 이유는 상태 백터의 상호작용은 기본적으로 시간에 따라 선형적이며 그 결과는 과거 몇몇 시간 단계에서 입력과 기본 RNN 구조처럼 크게 감쇄하지 않는 현재 출력을 연관시키는 경사이다. 이것은 LSTM이 RNN의 원래 체계보다 훨씬 더 효과적으로 장기적인 관계들을 학습할 수 있음을 의미한다.

> ### GRU(Gated Recurrent Units) - 2014년 요수아 벤지오의 그룹에서 제안한 LSTM유닛의 변형

##### LSTM 참고자료
https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
http://blog.varunajayasiri.com/numpy_lstm.html

##### 위 두 블로그를 비교해보면 Weight을 주는 방법이 다르다.
##### 첫번째는 기존의 RNN처럼 각 state 사이에 weight이 있는 반면
##### 두번째는 각 gate 마다 weight이 있어 기존 RNN의 weight의 개수보다 많다.
##### 첫번째 예시는 벡터의 차원이 맞지 않고 억지로 끼워 맞추더라도 학습이 되지가 않았다.
```python
def lossFun(inputs, targets, hprev, cprev):
    xs, hs, cs, is_, fs, os, gs, ys, ps= {}, {}, {}, {}, {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev) # t=0일때 t-1 시점의 hidden state가 필요하므로
    cs[-1] = np.copy(cprev)
    loss = 0
    H = hidden_size
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)) #(29, 1)
        xs[t][inputs[t]] = 1
        #print('t = ', t)
        # Wxh (100, 29), xs[t] (29, 1), Whh (100, 100), hs[t-1] (100, 1)
        #print(np.shape(Wxh))
        #print(np.shape(xs[t]))
        #print(np.shape(Whh))
        #print(np.shape(hs[t-1]))
        tmp = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh  # temp(100, 1)
        #print(np.shape(tmp))
        is_[t] = sigmoid(tmp[:H])
        fs[t] = sigmoid(tmp[:H])
        os[t] = sigmoid(tmp[:H])
        gs[t] = np.tanh(tmp[:H])
        cs[t] = fs[t] * cs[t-1] + is_[t] * gs[t]
        hs[t] = os[t] * np.tanh(cs[t])

    # compute loss
    #len(targets) : 25
    for i in range(len(targets)):
        idx = len(inputs) - len(targets) + i
        #print('idx = ', idx)
        ys[idx] = np.dot(Why, hs[idx]) + by  # unnormalized log probabilities for next chars
        #print(np.shape(ys[idx])) #(29, 1)
        #softmax
        ps[idx] = np.exp(ys[idx]) / np.sum(np.exp(ys[idx]))  # probabilities for next chars
        #cross-entropy error
        #print(np.shape(ps[idx])) #(29, 1)
        loss += -np.log(ps[idx][targets[i], 0])  # softmax (cross-entropy loss)

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext, dcnext = np.zeros_like(hs[0]), np.zeros_like(cs[0])
    n = 1
    a = len(targets) - 1
    #print('len(inputs)',len(inputs))
    for t in reversed(range(len(inputs))): #len(inputs) = 25, 25번 동안 W와 같은 변수들 계속 공유
        if n > len(targets):
            continue
        dy = np.copy(ps[t])
        dy[targets[a]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dc = dcnext + (1 - np.tanh(cs[t]) * np.tanh(cs[t])) * dh * os[t]  # backprop through tanh nonlinearity
        dcnext = dc * fs[t]
        di = dc * gs[t]
        df = dc * cs[t-1]
        do = dh * np.tanh(cs[t])
        dg = dc * is_[t]
        ddi = (1 - is_[t]) * is_[t] * di
        ddf = (1 - fs[t]) * fs[t] * df
        ddo = (1 - os[t]) * os[t] * do
        ddg = (1 - gs[t]**2) * dg
        #print('ddi sahpe = ', np.shape(ddi))
        #print('ddf sahpe = ', np.shape(ddf))
        #print('ddo sahpe = ', np.shape(ddo))
        #print('ddg sahpe = ', np.shape(ddg))
        #print('ddi ravel = ', np.shape(ddi.ravel()))
        #da = np.hstack((ddi.ravel(),ddf.ravel(),ddo.ravel(),ddg.ravel()))
        da = np.add(np.add(ddi.ravel(), ddf.ravel()), np.add(ddo.ravel(), ddg.ravel()))/4
        #print('da shpe = ', np.shape(da))
        dWxh += np.dot(da[:,np.newaxis],xs[t].T)
        dWhh += np.dot(da[:,np.newaxis],hs[t-1].T)
        dbh += da[:, np.newaxis]
        dhnext = np.dot(Whh.T, da[:, np.newaxis])
        n += 1
        a -= 1

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1], cs[len(inputs) - 1]
```

##### 두번째 예시는 확인 필요

##### 참고 자료
https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr
https://skymind.ai/kr/wiki/lstm#long
https://github.com/nicodjimenez/lstm/blob/master/lstm.py
http://docs.likejazz.com/lstm/
http://blog.varunajayasiri.com/numpy_lstm.html