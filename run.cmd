@ECHO OFF

python train.py -M NaiveConv1d  -NC 10 -E 20 -lr 1e-3
python train.py -M Naive4Conv1d -NC 4  -E 10 -lr 1e-4

python infer.py -M Naive4Conv1d -D SignalDataset --split train
python infer.py -M Naive4Conv1d -D SignalDataset --split test1
