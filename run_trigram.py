import torch
import torch.nn.functional as F


words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {c: i+1 for i, c in enumerate(chars)}
itos = {i+1: c for i, c in enumerate(chars)}
stoi['.'] = 0
itos[0] = '.'

# use same model
model = torch.nn.Sequential(
    torch.nn.Linear(54, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 27),
)
# load model weights 
model.load_state_dict(torch.load('trigram_checkpoint.pt'))
model.eval()

# turn off grad tracking when using 
with torch.no_grad():
    for _ in range(20):
        out = []
        ch1, ch2 = '.', '.'
        while True:
            x = torch.tensor([[stoi[ch1]], [stoi[ch2]]])
            xenc = F.one_hot(x, num_classes=27).float() # 27 elements 
            xenc = xenc.view(1, 54)

            logits = model(xenc)
            probs = F.softmax(logits, dim=1)

            ix = torch.multinomial(probs, num_samples=1).item()
            ch = itos[ix]

            if ch == '.':
                break

            out.append(ch)

            ch1, ch2 = ch2, ch
        out = ''.join(out)
        print(out)

            