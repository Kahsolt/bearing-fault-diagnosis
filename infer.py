#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

from train import *
from mk_data import DATA_FILES


@torch.inference_mode()
def run(args):
  ''' Model '''
  model: Model = globals()[args.model](4)
  state_dict = torch.load(LOG_PATH / f'{args.model}.pth')
  model.load_state_dict(state_dict)
  model = model.eval().to(device)

  ''' Data '''
  dataset_cls = globals()[args.dataset]
  testset = dataset_cls(args.split, transform=minmax_norm, n_class=4)
  testloader = DataLoader(testset, batch_size=1, shuffle=False)
  print('len(testset):', len(testset), 'len(testloader):', len(testloader))

  preds = []
  ok, tot = 0, 0
  for X, Y in tqdm(testloader):
    X = X.float().to(device)
    Y = Y.long().to(device)

    logits = model(X)
    pred = logits.argmax(dim=-1)
    preds.append(pred.item())

    ok += (pred == Y).sum().item()
    tot += len(X)

  if not args.split.startswith('test'):
    print(f'>> {args.split} accuracy: {ok / tot:.3%}')
  else:
    fp = LOG_PATH / f'submit_{args.model}.csv'
    print(f'>> writing {fp}...')
    with open(fp, 'w', encoding='utf-8') as fh:
      for p in preds:
        fh.write(f'{p}\n')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model',   default='NaiveConv1d',    choices=MODELS)
  parser.add_argument('-D', '--dataset', default='SignalDataset',  choices=DATASETS)
  parser.add_argument(      '--split',   default='test1', choices=['train', 'valid', 'test1', 'test2'])
  args = parser.parse_args()

  run(args)
