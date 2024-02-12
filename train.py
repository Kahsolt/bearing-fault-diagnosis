#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from argparse import ArgumentParser

from torch.optim import SGD, Adam, Adagrad, Adadelta, AdamW

from data import *
from model import *
from utils import *

MODELS = [
  'NaiveConv1d',
  'Naive4Conv1d',
  'SimpleConv1d',
  'SimpleConv2d',
  'MLP3d',
]
DATASETS = [
  'SignalDataset',
  'SpecDataset',
  'SignalPCADataset',
  'NaiveSignalDataset',
]


def run(args):
  ''' Model '''  
  model: Model = globals()[args.model](args.n_class)
  if hasattr(model, 'base_cls'):
    print(f'>> load base model: {model.base_cls.__name__}')
    state_dict = torch.load(LOG_PATH / f'{model.base_cls.__name__}.pth')
    model.load_weights(state_dict)
  print(model)
  print('param_cnt:', sum([p.numel() for p in model.parameters() if p.requires_grad]))

  ''' Data '''
  dataset_cls = globals()[args.dataset]
  trainset = dataset_cls('train', transform=minmax_norm, n_class=args.n_class)
  trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
  validset = dataset_cls('valid', transform=minmax_norm, n_class=args.n_class)
  validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)
  print('len(trainset):', len(trainset), 'len(trainloader):', len(trainloader))
  print('len(validset):', len(validset), 'len(validloader):', len(validloader))

  ''' Ckpt '''  
  fp = LOG_PATH / f'{args.model}.pth'
  if not 'from pretrained':
    try:
      print(f'>> load model ckpt from {fp}...')
      state_dict = torch.load(fp)
      model.load_state_dict(state_dict)
    except: pass
  model = model.to(device)

  ''' Optim '''
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

  ''' Train '''
  step = 0
  best_acc = 0
  for epoch in range(args.epochs):
    ok, tot = 0, 0
    model.train()
    for X, Y in trainloader:
      X = X.float().to(device)
      Y = Y.long().to(device)

      optimizer.zero_grad()
      logits = model(X)
      loss = criterion(logits, Y)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        pred = logits.argmax(dim=-1)
        ok += (pred == Y).sum().item()
        tot += len(X)

      step += 1

      if step % 20 == 0:
        print(f'>> [step {step}] loss: {loss.item()}, acc: {ok / tot:.3%}')

    print(f'>> [Epoch: {epoch + 1}/{args.epochs}] train accuracy: {ok / tot:.3%}')

    with torch.inference_mode():
      ok, tot = 0, 0
      model.eval()
      for X, Y in validloader:
        X = X.float().to(device)
        Y = Y.long().to(device)

        logits = model(X)
        pred = logits.argmax(dim=-1)
        ok += (pred == Y).sum().item()
        tot += len(X)

      acc = ok / tot
      print(f'>> [Epoch: {epoch + 1}/{args.epochs}] valid accuracy: {acc:.3%}')

      if acc > best_acc:
        best_acc = acc
        print(f'>> save new best to {fp}')
        torch.save(model.state_dict(), fp)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M',  '--model',      default='NaiveConv1d',   choices=MODELS)
  parser.add_argument('-D',  '--dataset',    default='SignalDataset', choices=DATASETS)
  parser.add_argument('-NC', '--n_class',    default=4,    type=int,  choices=[4, 10])
  parser.add_argument('-E',  '--epochs',     default=20,   type=int)
  parser.add_argument('-B',  '--batch_size', default=20,   type=int)
  parser.add_argument('-lr', '--lr',         default=1e-3, type=eval)
  args = parser.parse_args()

  run(args)
