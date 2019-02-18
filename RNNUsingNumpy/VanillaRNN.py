# coding: utf-8
'''
Created on 2019. 2. 16.

@author: Insup Jung
'''

import numpy as np

data = open('data/input.txt', 'r').read().lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d chracters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}
print(char_to_ix)
print(ix_to_char)

hidden_size = 100
seq_length = 25
learning_rate = 1e-1

Wxh = np.random.randn(hidden_size, vocab_size)*0.01
Whh = np.random.randn(hidden_size, hidden_size)*0.01
Why = np.random.randn(vocab_size, hidden_size)*0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))
print('Wxh shape = ', np.shape(Wxh))
print('Whh shape = ', np.shape(Whh))
print('Why shape = ', np.shape(Why))

def lossFun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {} #dictionary 선언, 각 state를 dictionary로 저장
    hs[-1] = np.copy(hprev)
    loss = 0
    
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)) #input_state        
        xs[t][inputs[t]] = 1 # 각 단어별로 one-hot encoding
        
        #hidden_state        
        #(Wxh X 단어벡터) + (Whh X 전 단계의 hidden_state) + bias        
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        
        #Softmax 함수
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        
        #targets[t]는 레이블인데 정답만 1이고 나머지는 0이므로 아래와 같이 구할 수 있다.
        loss += -np.log(ps[t][targets[t], 0])
    #print(xs)
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    
    #backpropagation
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_index, n):
    x = np.zeros((vocab_size, 1))
    x[seed_index] = 1
    result = [] #result
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh) #shape = (hidden_size, 1)
        y = np.dot(Why, h) + by #shape = (words_count, 1)
        p = np.exp(y) / np.sum(np.exp(y)) #shape = (words_count, 1)
        ix = np.random.choice(range(vocab_size), p=p.ravel()) # p parameter는 probabilities를 제공한다.
        #print(ix)
        #다음 index를 만들어준다.
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        # 결과를 만들어준다.
        result.append(ix)
    return result

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
i = 0
while True:
    #i += 1
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1))
        p = 0 
        # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]] #input data
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]] #input data에서 index가 1씩 뒤로 밀린 target data
    #print(targets)
    
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print ('----\n %s \n----' % (txt, ))
    
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    # loss가 천천히 되도록 해주는 장치인 것 같다.
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
    #if n % 100 == 0: print ('abc') # print progress
        
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):        
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    
    p += seq_length # move data pointer
    n += 1 # iteration counter 