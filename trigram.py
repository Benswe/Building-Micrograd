import __NN__ as nn

data = open('names.txt', 'w').readlines()

# prepare training examples, 




# build stoi in order to index all lowercase letters
chars = sorted(list(set(data)))

itos = {i+1:s for i, s in enumerate(chars)} # 1:a 2:b 3:c etc

