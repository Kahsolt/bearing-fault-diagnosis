#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from train import *


DATASET_CLS = {
  'SimpleConv1d': SignalTestDataset,
  'SimpleConv2d': SpecTestDataset,
  'MLP3d': SignalPCATestDataset,
}


@torch.inference_mode()
def run(args):
  model = globals()[args.model]()
  state_dict = torch.load(LOG_PATH / f'{args.model}.pth')
  model.load_state_dict(state_dict)
  model = model.eval().to(device)

  dataset_cls = DATASET_CLS[args.model]
  testset = dataset_cls('test1', transform=minmax_norm)
  testloader = DataLoader(testset, batch_size=1, shuffle=False)
  print('len(testset):', len(testset), 'len(testloader):', len(testloader))

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
  run(get_args())
