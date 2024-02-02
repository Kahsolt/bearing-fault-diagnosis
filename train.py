#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from torch.optim import SGD, Adam, Adagrad, Adadelta, AdamW

from data import SignalTrainDataset, DataLoader
from model import SimpleConv1d

from utils import *

EPOCH = 10
BATCH_SIZE = 20
LR = 0.001


def run():
  model = SimpleConv1d().to(device)
  print(model)
  print('param_cnt:', sum([p.numel() for p in model.parameters()]))

  if not 'from pretrained':
    try:
      print('>> load model ckpt...')
      state_dict = torch.load(MODEL_PATH)
      model.load_state_dict(state_dict)
    except: pass

  trainset = SignalTrainDataset('train', transform=minmax_norm)
  trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
  validset = SignalTrainDataset('valid', transform=minmax_norm)
  validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
  print('len(trainset):', len(trainset), 'len(trainloader):', len(trainloader))
  print('len(validset):', len(validset), 'len(validloader):', len(validloader))

  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=LR, weight_decay=5e-4)
  #optimizer = SGD(model.parameters(), lr=LR, momentum=0.9)

  step = 0
  best_acc = 0
  for epoch in range(EPOCH):
    model.train()
    for X, Y in trainloader:
      X = X.float().to(device)
      Y = Y.long().to(device)

      optimizer.zero_grad()
      logits = model(X)
      loss = criterion(logits, Y)
      loss.backward()
      optimizer.step()

      step += 1

      if step % 20 == 0:
        print(f'>> [step {step}] loss: {loss.item()}')

    ok, tot = 0, 0
    with torch.inference_mode():
      model.eval()
      for X, Y in validloader:
        X = X.float().to(device)
        Y = Y.long().to(device)

        logits = model(X)
        pred = logits.argmax(dim=-1)
        ok += (pred == Y).sum().item()
        tot += len(X)

      acc = ok / tot
      print(f'>> [Epoch: {epoch + 1}/{EPOCH}] accuracy: {acc:.3%}')

      if acc > best_acc:
        best_acc = acc
        print(f'>> save new best to {MODEL_PATH}')
        torch.save(model.state_dict(), MODEL_PATH)


if __name__ == '__main__':
  run()
