# Experiment notes: Pilot

Training runs:

---- WITH TRIGGER PREPROC---
01. Rb CUDA_VISIBLE_DEVICES=0 python main.py --task triplet --mode train --bert_model_path "roberta-base" --bert_tokenizer "roberta-base" --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200

RUN1:
best epoch: 175 best dev triplet f1: 0.23155

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.58709       R:0.61245       F1:0.59950
Opinion term    P:0.44103       R:0.38782       F1:0.41272
triplet P:0.25701       R:0.20780       F1:0.22980

RUN2:
best epoch: 141 best dev triplet f1: 0.24540 p: 0.24074 r: 0.25025

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.56693       R:0.54892       F1:0.55778
Opinion term    P:0.41292       R:0.42503       F1:0.41889
triplet P:0.23718       R:0.20145       F1:0.21786

02. BEST Rb CUDA_VISIBLE_DEVICES=2 python main.py --task triplet --mode train --bert_model_path "roberta-base" --bert_tokenizer "roberta-base" --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 2

RUN 1:

best epoch: 99  best dev triplet f1: 0.23486

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.57888       R:0.60610       F1:0.59218
Opinion term    P:0.41260       R:0.39121       F1:0.40162
triplet P:0.26352       R:0.20780       F1:0.23237

RUN 2:
best epoch: 196 best dev triplet f1: 0.23783 p: 0.24066 r: 0.23506

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.58427       R:0.59466       F1:0.58942
Opinion term    P:0.40511       R:0.37542       F1:0.38970
triplet P:0.21616       R:0.18693       F1:0.20049

03. Rb CUDA_VISIBLE_DEVICES=0 python main.py --task triplet --mode train --bert_model_path "roberta-base" --bert_tokenizer "roberta-base" --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 3

RUN1:
best epoch: 57  best dev triplet f1: 0.23166

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.59849       R:0.60610       F1:0.60227
Opinion term    P:0.45681       R:0.35175       F1:0.39745
triplet P:0.28610       R:0.19238       F1:0.23006

RUN2:
best epoch: 128 best dev triplet f1: 0.23342 p: 0.22986 r: 0.23708

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.56851       R:0.60102       F1:0.58431
Opinion term    P:0.41062       R:0.42728       F1:0.41878
triplet P:0.24724       R:0.22323       F1:0.23462

04. BEST ROBERTA LARGE Rl CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --bert_model_path "roberta-large" --bert_tokenizer "roberta-large" --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --bert_feature_dim 1024

RUN 1:
best epoch: 113 best dev triplet f1: 0.22912

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.56209       R:0.54638       F1:0.55412
Opinion term    P:0.39954       R:0.39233       F1:0.39590
triplet P:0.20887       R:0.17514       F1:0.19052

RUN 2:
best epoch: 107 best dev triplet f1: 0.23973 p: 0.26503 r: 0.21884

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.61826       R:0.56798       F1:0.59205
Opinion term    P:0.44305       R:0.39910       F1:0.41993
triplet P:0.26087       R:0.18512       F1:0.21656


05.  Rl CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --bert_model_path "roberta-large" --bert_tokenizer "roberta-large" --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --bert_feature_dim 1024 --nhops 2

RUN 1:
best epoch: 113 best dev triplet f1: 0.24268

Aspect term     P:0.59038       R:0.57687       F1:0.58355
Opinion term    P:0.43632       R:0.39008       F1:0.41190
triplet P:0.24421       R:0.19147       F1:0.21465

RUN 2:
best epoch: 131 best dev triplet f1: 0.23776 p: 0.23947 r: 0.23607

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.59853       R:0.62135       F1:0.60973
Opinion term    P:0.42674       R:0.41375       F1:0.42015
triplet P:0.25206       R:0.22232       F1:0.23626

06. FinRoBERTa CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --bert_model_path "abhilash1910/financial_roberta" --bert_tokenizer "abhilash1910/financial_roberta"  --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200

RUN 1:
best epoch: 112 best dev triplet f1: 0.09634 p: 0.09533 r: 0.09736

Evaluation on testset:
Sentence wmt07:11 truncated due to max_sequence_len 100.
Aspect term     P:0.31177       R:0.44091       F1:0.36526
Opinion term    P:0.20167       R:0.24464       F1:0.22109
triplet P:0.08616       R:0.08076       F1:0.08337


07. FinRoBERTa CUDA_VISIBLE_DEVICES=3 python main.py --task triplet --mode train --bert_model_path "abhilash1910/financial_roberta" --bert_tokenizer "abhilash1910/financial_roberta"  --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 2

RUN 1:
train1

08. ProsusAI/FinBERT-sent CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --bert_model_path "pretrained/finbertphrasebanksent" --bert_tokenizer "pretrained/finbertphrasebanksent"  --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200

--------
===== WITH EMPTY INSTANCES =====

01. Rb CUDA_VISIBLE_DEVICES=3 python main.py --task triplet --mode train --bert_model_path "roberta-base" --bert_tokenizer "roberta-base" --dataset sentivent-event-devproto --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200

RUN 1:
best epoch: 161 best dev triplet f1: 0.21754 p: 0.24837 r: 0.19352

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.59937       R:0.48285       F1:0.53483
Opinion term    P:0.40584       R:0.31342       F1:0.35369
triplet P:0.25668       R:0.17423       F1:0.20757

02. Rb CUDA_VISIBLE_DEVICES=3 python main.py --task triplet --mode train --bert_model_path "roberta-base" --bert_tokenizer "roberta-base" --dataset sentivent-event-devproto --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 2

RUN 1:
best epoch: 135 best dev triplet f1: 0.21565 p: 0.23452 r: 0.19959

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.55076       R:0.50318       F1:0.52590
Opinion term    P:0.41757       R:0.34837       F1:0.37984
triplet P:0.23781       R:0.18149       F1:0.20587

03. Rl CUDA_VISIBLE_DEVICES=2 python main.py --task triplet --mode train --bert_model_path "roberta-large" --bert_tokenizer "roberta-large" --dataset sentivent-event-devproto --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --bert_feature_dim 1024

RUN 1:
train2

----------
---JOINED-SEMEVAL
01. Rb CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --bert_model_path "roberta-base" --bert_tokenizer "roberta-base" --dataset joinedsemeval --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200

best epoch: 74  best dev triplet f1: 0.76729 p: 0.78078 r: 0.75425

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.85746       R:0.87998       F1:0.86857
Opinion term    P:0.87632       R:0.87981       F1:0.87806
triplet P:0.74960       R:0.74076       F1:0.74515

02. Rl CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --bert_model_path "roberta-large" --bert_tokenizer "roberta-large" --dataset joinedsemeval --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --bert_feature_dim 1024

best epoch: 84  best dev triplet f1: 0.77370 p: 0.79059 r: 0.75752

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.84672       R:0.89312       F1:0.86930
Opinion term    P:0.86574       R:0.88334       F1:0.87445
triplet P:0.74832       R:0.74390       F1:0.74610




---------------------------------------
---BEFORE TRIGGER PREPROC
01. BEST --task triplet --mode train --bert_model_path "roberta-base" --bert_tokenizer "roberta-base" --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 2

best epoch: 131	best dev triplet f1: 0.23815

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term	P:0.60305	R:0.58519	F1:0.59398
Opinion term	P:0.44938	R:0.41791	F1:0.43308
triplet	P:0.26585	R:0.21377	F1:0.23699


02. CUDA_VISIBLE_DEVICES=0 python main.py --task triplet --mode train --bert_model_path "roberta-large" --bert_tokenizer "roberta-large"  --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 3 --bert_feature_dim 1024

best epoch: 140 best dev triplet f1: 0.23333

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.60180       R:0.57654       F1:0.58890
Opinion term    P:0.41646       R:0.38921       F1:0.40237
triplet P:0.23661       R:0.18962       F1:0.21053


03. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --bert_model_path "roberta-base" --bert_tokenizer "roberta-base"  --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200

best epoch: 164 best dev triplet f1: 0.23536

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.61294       R:0.59630       F1:0.60451
Opinion term    P:0.43704       R:0.40643       F1:0.42118
triplet P:0.25905       R:0.20483       F1:0.22877

04. CUDA_VISIBLE_DEVICES=0 python main.py --task triplet --mode train --bert_model_path "roberta-large" --bert_tokenizer "roberta-large"  --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 2 --bert_feature_dim 1024

best epoch: 31  best dev triplet f1: 0.23704

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.61294       R:0.59630       F1:0.60451
Opinion term    P:0.43704       R:0.40643       F1:0.42118
triplet P:0.25905       R:0.20483       F1:0.22877

05. FinRoBERTa doesn't work CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --bert_model_path "abhilash1910/financial_roberta" --bert_tokenizer "abhilash1910/financial_roberta"  --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200

best epoch: 133 best dev triplet f1: 0.09256

Evaluation on testset:
Sentence wmt07:11 truncated due to max_sequence_len 100.
Aspect term     P:0.05882       R:0.04938       F1:0.05369
Opinion term    P:0.03886       R:0.01722       F1:0.02387
triplet P:0.00361       R:0.00089       F1:0.00143

06. CUDA_VISIBLE_DEVICES=0 python main.py --task triplet --mode train --bert_model_path "roberta-large" --bert_tokenizer "roberta-large"  --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --bert_feature_dim 1024

best epoch: 144 best dev triplet f1: 0.23282

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.60853       R:0.58148       F1:0.59470
Opinion term    P:0.43017       R:0.39610       F1:0.41243
triplet P:0.25891       R:0.19499       F1:0.22245


07. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --bert_model_path "roberta-large" --bert_tokenizer "roberta-large" --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --bert_feature_dim 1024 --nhops 4

best epoch: 67  best dev triplet f1: 0.22439

Evaluation on testset:
Ignored unknown kwargs option trim_offsets
Aspect term     P:0.60853       R:0.58148       F1:0.59470
Opinion term    P:0.43017       R:0.39610       F1:0.41243
triplet P:0.25891       R:0.19499       F1:0.22245

----
2. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --epochs 200 --batch 16

best epoch: 32  best dev triplet f1: 0.19888

Evaluation on testset:
Aspect term     P:0.56210       R:0.54198       F1:0.55185
Opinion term    P:0.36223       R:0.35017       F1:0.35610
triplet P:0.21447       R:0.15116       F1:0.17733

3. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200

best epoch: 162 best dev triplet f1: 0.19665

Evaluation on testset:
Aspect term     P:0.53176       R:0.55802       F1:0.54458
Opinion term    P:0.38797       R:0.37773       F1:0.38278
triplet P:0.20198       R:0.16458       F1:0.18137


4. CUDA_VISIBLE_DEVICES=0 python main.py --task triplet --mode train --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 2

best epoch: 197 best dev triplet f1: 0.20534

Evaluation on testset:
Aspect term     P:0.54919       R:0.54444       F1:0.54681
Opinion term    P:0.38219       R:0.38921       F1:0.38567
triplet P:0.21231       R:0.17889       F1:0.19417


5. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 3

best epoch: 95  best dev triplet f1: 0.19014

Evaluation on testset:
Aspect term     P:0.52719       R:0.55062       F1:0.53865
Opinion term    P:0.39438       R:0.40299       F1:0.39864
triplet P:0.20000       R:0.16995       F1:0.18375

Pair1. CUDA_VISIBLE_DEVICES=1 python main.py --task pair --mode train --dataset sentivent-event-devproto-no-empty --max_sequence_len 100 --epochs 100 --batch 12

best epoch: 97  best dev pair f1: 0.25636

Evaluation on testset:
Aspect term     P:0.54601       R:0.54938       F1:0.54769
Opinion term    P:0.38710       R:0.37199       F1:0.37939
pair    P:0.26531       R:0.23256       F1:0.24786

-----
--max_sequence_len 100 -> OOM

1. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 80

best epoch: 81	best dev triplet f1: 0.14527

Evaluation on testset:
Aspect term	P:0.54751	R:0.56279	F1:0.55505
Opinion term	P:0.28881	R:0.34783	F1:0.31558
triplet	P:0.16129	R:0.15326	F1:0.15717

2. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 90

best epoch: 95  best dev triplet f1: 0.14944

Evaluation on testset:
Aspect term     P:0.54585       R:0.58140       F1:0.56306
Opinion term    P:0.24752       R:0.32609       F1:0.28143
triplet P:0.13380       R:0.14559       F1:0.13945

3. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --batch_size 16

best epoch: 71  best dev triplet f1: 0.15789

Evaluation on testset:
Aspect term     P:0.56422       R:0.57209       F1:0.56813
Opinion term    P:0.28346       R:0.31304       F1:0.29752
triplet P:0.17826       R:0.15709       F1:0.16701

4. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --batch_size 16 --learning_rate 1e-5 --epochs 200

best epoch: 105 best dev triplet f1: 0.13226

Evaluation on testset:
Aspect term     P:0.49412       R:0.58605       F1:0.53617
Opinion term    P:0.29368       R:0.34348       F1:0.31663
triplet P:0.13971       R:0.14559       F1:0.14259

5. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --batch_size 16 --learning_rate 2e-5 --epochs 200

best epoch: 191 best dev triplet f1: 0.16456

Evaluation on testset:
Aspect term     P:0.52675       R:0.59535       F1:0.55895
Opinion term    P:0.29592       R:0.37826       F1:0.33206
triplet P:0.14626       R:0.16475       F1:0.15495

6. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200

best epoch: 80  best dev triplet f1: 0.17391

Evaluation on testset:
Aspect term     P:0.54959       R:0.61860       F1:0.58206
Opinion term    P:0.29110       R:0.36957       F1:0.32567
triplet P:0.15194       R:0.16475       F1:0.15809

7. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --batch_size 6 --learning_rate 1e-5 --epochs 200

best epoch: 144 best dev triplet f1: 0.16584


Evaluation on testset:
Aspect term     P:0.53689       R:0.60930       F1:0.57081
Opinion term    P:0.26897       R:0.33913       F1:0.30000
triplet P:0.12903       R:0.13793       F1:0.13333

8.  CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --batch_size 6 --learning_rate 2e-5 --epochs 200

best epoch: 48  best dev triplet f1: 0.15537


Evaluation on testset:
Aspect term     P:0.50195       R:0.60000       F1:0.54661
Opinion term    P:0.31034       R:0.35217       F1:0.32994
triplet P:0.13962       R:0.14176       F1:0.14068

9. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 1e-5 --epochs 200 

best epoch: 196 best dev triplet f1: 0.16014

Evaluation on testset:
Aspect term     P:0.51793       R:0.60465       F1:0.55794
Opinion term    P:0.29885       R:0.33913       F1:0.31772
triplet P:0.16538       R:0.16475       F1:0.16507

10. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 -nhops 2

best epoch: 119 best dev triplet f1: 0.16867


Evaluation on testset:
Aspect term     P:0.53604       R:0.55349       F1:0.54462
Opinion term    P:0.26740       R:0.31739       F1:0.29026
triplet P:0.16049       R:0.14943       F1:0.15476

11. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --batch_size 8 --learning_rate 2e-5 --epochs 200 --nhops 3

Aspect term     P:0.56667       R:0.56432       F1:0.56549
Opinion term    P:0.27181       R:0.28622       F1:0.27883
triplet P:0.17623       R:0.14238       F1:0.15751

best epoch: 75  best dev triplet f1: 0.16118


Evaluation on testset:
Aspect term     P:0.50000       R:0.57209       F1:0.53362
Opinion term    P:0.26736       R:0.33478       F1:0.29730
triplet P:0.15909       R:0.16092       F1:0.16000

11. CUDA_VISIBLE_DEVICES=1 python main.py --task triplet --mode train --dataset sentivent-devproto-no-empty --max_sequence_len 100 --epochs 100 --nhops 2 --batch_size 12

best epoch: 12  best dev triplet f1: 0.14508

Evaluation on testset:
Aspect term     P:0.50209       R:0.55814       F1:0.52863
Opinion term    P:0.26493       R:0.30870       F1:0.28514
triplet P:0.12796       R:0.10345       F1:0.11441
