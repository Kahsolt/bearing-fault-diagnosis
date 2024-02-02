#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from data import SignalTestDataset, DataLoader
from model import SimpleConv1d

from utils import *


@torch.inference_mode()
def run():
  model = SimpleConv1d()
  state_dict = torch.load(MODEL_PATH)
  model.load_state_dict(state_dict)
  model = model.eval().to(device)

  testset = SignalTestDataset('test1', transform=minmax_norm)
  testloader = DataLoader(testset, batch_size=1, shuffle=False)

  preds = []
  for X in tqdm(testloader):
    X = X.float().to(device)

    logits = model(X)
    pred = logits.argmax(dim=-1)
    preds.append(pred.item())

  print(f'>> writing {SUBMIT_PATH}...')
  with open(SUBMIT_PATH, 'w', encoding='utf-8') as fh:
    for p in preds:
      fh.write(f'{p}\n')


if __name__ == '__main__':
  run()
