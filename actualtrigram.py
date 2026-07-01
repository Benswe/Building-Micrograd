import torch
import torch.nn.functional as F


words = open('names.txt', 'r').read().splitlines()

# build itos and stoi

chars = sorted(list(set(''.join(words))))


stoi = {c: i+1 for i,c in enumerate(chars)}
itos = {i+1: c for i,c in enumerate(chars)}
stoi['.'] = 0 # '.' represents start of word or end of word 
itos[0] = '.'

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

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs).float()
# reshape for neural net input
xenc = xenc.view(xs.shape[0], 54)
nums = xs.nelement()/2

model = torch.nn.Sequential(
    torch.nn.Linear(54, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 27),
    )

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
for step in range(100):
    # forward pass
    logits = model(xenc)
    loss = F.cross_entropy(logits, ys)

    # backwards pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())


torch.save(model.state_dict(), 'trigram_checkpoint.pt')