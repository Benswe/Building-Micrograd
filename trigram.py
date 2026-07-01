import __NN__ as nn
import random

# bring our dataset that we will train and test on in 

data = open('names.txt', 'r').read().splitlines()

# all lowercase characters
chars = sorted(list(set(''.join(data))))

# decoding
itos = {i+1:s for i, s in enumerate(chars)} # 1:a 2:b 3:c etc
# encoding 
stoi = {s: i+1 for i,s in enumerate(chars)}
itos[0] = '.'
stoi['.'] = 0

# prepare trigram training examples


# build a vector of length 54, where the first 27 correspond to ch1
# second 27 correspond to ch2
def one_hot_pair(ch1, ch2):
    x = [0] * 54
    # use stoi to encode char as index 
    x[stoi[ch1]] = 1
    x[27 + stoi[ch2]] = 1
    return x

xs, ys = [], [] # xs are inputs, ys are targets 
for name in data:
    chars = ['.'] + ['.'] + list(name) + ['.']
    for ch1, ch2, target in zip(chars, chars[1:], chars[2:]):
        xs.append(one_hot_pair(ch1, ch2)) # returns a list
        ys.append(stoi[target])

# input vector xs of size 54, output a probability distribution of size 27 at end
model = nn.MLP(54, [32, 32, 27])

def softmax(logits):
    # use .exp() from __engine__.py
    exps = [logit.exp() for logit in logits]
    total = sum(exps)
    return [e/total for e in exps]

batch_size = 32
for step in range(1, 2001):
    ix = random.sample(range(len(xs)), batch_size)
    xb = [xs[i] for i in ix]
    yb = [ys[i] for i in ix]
    # forward pass on all the one-hot encodings
    logits = [model(x) for x in xb]
    probs = [softmax(logit) for logit in logits]

    loss = sum([-prob[y].log() for prob, y in zip(probs, yb)])/len(yb)
    if step % 100 == 0:
        print(step, loss)

    # backward pass
    loss.backward()

    for w in model.parameters():
        w.data += -0.01 * w.grad #gd
        # zero out the gradient
        w.grad = 0


    