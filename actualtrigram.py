import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

# build itos and stoi

chars = sorted(list(set(''.join(words))))


stoi = {c: i+1 for i,c in enumerate(chars)}
itos = {i+1: c for i,c in enumerate(chars)}
stoi['.'] = 0 # '.' represents start of word or end of word 
itos[0] = '.'

def generate_datasets(words):
    xs = []
    ys = []
    for w in words:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, target in zip(chs, chs[1:], chs[2:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            idx3 = stoi[target]
            # populate the train and test sets
            xs.append((idx1, idx2))
            ys.append(idx3)

    X = torch.tensor(xs)
    Y = torch.tensor(ys)
    return X, Y

random.seed(2147483647)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

xtr, ytr = generate_datasets(words[:n1]) # 80%
xval, yval = generate_datasets(words[n1:n2]) # 10%
xtest, ytest = generate_datasets(words[n2:]) # 10%

print(f"train: {xtr.shape[0]} examples")
print(f"val:   {xval.shape[0]} examples")
print(f"test:  {xtest.shape[0]} examples")


# start the weights/biases very low so the initial ypreds are uniform, thus the loss will be must lower (-log(1/27))
W1 = (torch.randn(size=(54, 150)) * 0.2).requires_grad_()
b1 = (torch.randn((150)) * 0.01).requires_grad_()
W2 = (torch.randn(size=(150, 27)) * 0.01).requires_grad_()
b2 = (torch.randn((27)) * 0).requires_grad_()

parameters = [W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))
batch_size = 32
# stats
lossi = []
stepi = []

def encode(X):
    xenc = F.one_hot(X, 27).float()
    return xenc.view(xenc.shape[0], 54)


def forward(X):
    h = torch.tanh(encode(X) @ W1 + b1)
    return h @ W2 + b2

for i in range(30000):

    # for batching
    idx = torch.randint(0, xtr.shape[0], (batch_size, ))
    xb = xtr[idx]
    yb = ytr[idx]

    logits = forward(xb)
    

    for p in parameters:
        p.grad = None

    loss = F.cross_entropy(logits, yb)
    loss.backward()
    if i == 0:
        print(f"Initial loss: {loss.item()}")
    lossi.append(loss.item())
    stepi.append(i)

    for p in parameters:
        p.data += -0.1 * p.grad

@torch.no_grad()
def split_loss(split):
    x, y ={
        "train": (xtr, ytr),
        "val": (xval, yval),
        "test": (xtest, ytest),
    }[split]

    logits = forward(x)

    loss = F.cross_entropy(logits, y)
    print(f"{split} loss: {loss.item()}")

split_loss("train")
split_loss("val")
split_loss("test")

plt.plot(stepi, lossi)
plt.show()