@ECHO OFF

REM set "train" or "refine"
SET RECIPE=%1


IF "%RECIPE%"=="" SET RECIPE=train
GOTO %RECIPE%

REM derectly train Y4
:train
python train.py -M NaiveConv1d -D SignalDataset -NC 4 -E 40 -lr 1e-4
python infer.py -M NaiveConv1d -D SignalDataset --split train
python infer.py -M NaiveConv1d -D SignalDataset --split test1
GOTO EOF

REM pretrain Y10 to refine Y4
:refine
python train.py -M NaiveConv1d  -D NaiveSignalDataset -NC 10 -E 10 -lr 1e-3
python train.py -M Naive4Conv1d -D SignalDataset      -NC 4  -E 20 -lr 1e-4
python infer.py -M Naive4Conv1d -D SignalDataset --split train
python infer.py -M Naive4Conv1d -D SignalDataset --split test1
GOTO EOF

:EOF
