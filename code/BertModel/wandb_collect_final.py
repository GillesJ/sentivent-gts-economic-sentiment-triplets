#!/usr/bin/env python3
"""
A. Collect devset hyperparameter optimization search from WandB.ai and select winning devset architecture.
B. Retrain on holdin set (devset + dev eval set) and test on holdout 5 times and average as in original GTS paper.
If holdout results file already exist, do not retrain, but load from file.
C. Load results, compute McNemar's significance test on holdout testset predictions and markup LaTeX table.

For paper experiments, preds were logged at wandb.com and downloaded to winpreds subdir in
../../data/{experiment-task}-results-hyperparameter-search/ which also contains hyperopt results
per architecture.

wandb_collect_final.py in sentivent-implicit-economic-sentiment
7/20/21 Copyright (c) Gilles Jacobs
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    accuracy_score,
)
from mlxtend.evaluate import mcnemar_table, mcnemar, cochrans_q
from ast import literal_eval
import json, os
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import trange

from data import load_data_instances, DataIterator
from model import MultiInferBert
import utils
import wandb
from accelerate import Accelerator

from pathlib import Path


def retrain_test():
    global trainset, testset
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    model = MultiInferBert(args).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': args.learning_rate},
        {'params': model.cls_linear.parameters()}
    ], lr=args.learning_rate)

    model, optimizer, trainset = accelerator.prepare(model, optimizer, trainset)

    res = {}
    patience = 0
    res["best_joint_f1"] = -1
    res["best_joint_epoch"] = -1
    print(f'Train epochs {args.epochs}, patience: {args.patience}, warmup patience: {args.patience_init}')
    for i in range(args.epochs):
        print(f'Epoch:{i}/{args.epochs} (patience={patience}/{args.patience})')
        for j in trange(trainset.batch_count):
            _, tokens, lengths, masks, _, _, aspect_tags, tags = trainset.get_batch(j)
            preds = model(tokens, masks)

            preds_flatten = preds.reshape([-1, preds.shape[3]])
            tags_flatten = tags.reshape([-1])
            loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        res["joint_precision"], res["joint_recall"], res["joint_f1"], prf_a, prf_o = eval(model, testset)
        res["aspect_precision"], res["aspect_recall"], res["aspect_f1"] = prf_a
        res["opinion_precision"], res["opinion_recall"], res["opinion_f1"] = prf_o


        if res["joint_f1"] > res["best_joint_f1"]:
            patience = 0
            res["best_joint_f1"] = res["joint_f1"]
            res["best_joint_epoch"] = i
            res["best_joint_precision"] = res["joint_precision"]
            res["best_joint_recall"] = res["joint_recall"]
            res["best_aspect_precision"], res["best_aspect_recall"], res["best_aspect_f1"] = res["aspect_precision"], res["aspect_recall"], res["aspect_f1"]
            res["best_opinion_precision"], res["best_opinion_recall"], res["best_opinion_f1"] = res["opinion_precision"], res["opinion_recall"], res["opinion_f1"]
            res["best_preds"] = [x.sentence_pack for x in testset.instances]

        else: # no improvement so increment patience
            patience += 1
        if patience == args.patience:
            print(f"Patience of {args.patience} exceeded, stopping run.")
            break
        if i == args.patience_init and res["best_joint_f1"] <= 0:
            print(f"Initial patience of {args.patience_init} exceeded, stopping run.")
            break
    print('Results final epoch {}\t {} f1: {:.5f} p: {:.5f} r: {:.5f}\n\n'.format(args.epochs,
                                                                                  args.task,
                                                                                  res["joint_f1"],
                                                                                  res["joint_precision"],
                                                                                  res["joint_recall"]))
    print('Results best epoch {}\t {} f1: {:.5f} p: {:.5f} r: {:.5f}\n\n'.format(res["best_joint_epoch"],
                                                                                  args.task,
                                                                                  res["best_joint_f1"],
                                                                                  res["best_joint_precision"],
                                                                                  res["best_joint_recall"]))

    # testset for manual inspection
    res["preds"] = [x.sentence_pack for x in testset.instances]

    return res

def eval(model, dataset):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        for i in range(dataset.batch_count):
            sentence_ids, tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags = dataset.get_batch(i)
            preds = model(tokens, masks)
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_labels.append(tags)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)
            all_ids.extend(sentence_ids)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = utils.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        precision, recall, f1 = metric.score_uniontags(dataset)
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()

        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))
        print(args.task + '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1, aspect_results, opinion_results

def post_process_latex(tab, n_step=3, len_header=7, len_footer=3):
    '''

    :param tab: table as DataFrame
    :param n_step: amount of ablations per model arx
    :param len_header: len lines of header in latex tabkle, 7 with label and caption
    :param len_footer: len line of footer in latex table, 3 is default table footer
    :return: str with cleaned latex table
    '''
    lines = tab.splitlines()
    markup_remove = [
        "\\centering",
        #  "\\toprule",
        "\\midrule",  # midrule gets placed wrongly
    ]
    lines = [l for l in lines if l not in markup_remove]  # remove some markup
    # add grouping hspace every third line of content
    n = n_step  # amount steps to vertical space n ablations per arx
    len_header = len_header - len(
        markup_remove
    )
    for i in range(n + len_header, len(lines) - len_footer - n, n):
        lines[i] = lines[i].replace("\\\\", "\\\\[5pt]")
    tab_proc = "\n".join(lines)

    return tab_proc

def test_mcnemar(y_target, y_model1, y_model2):
    """
    https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
    """
    assert y_target.shape == y_model1.shape == y_model2.shape

    cont_table = mcnemar_table(
        y_target=y_target,
        y_model1=y_model1,
        y_model2=y_model2,
    )
    # print(pd.DataFrame(cont_table, columns=['model2 correct', 'model2 wrong'], index=['model1 correct', 'model1 wrong']))
    chi2, p = mcnemar(ary=cont_table, corrected=True)

    # APA (American Psychological Association) style, which shows three digits but omits the leading zero (.123).
    # P values less than 0.001 shown as "< .001". All P values less than 0.001 are summarized with three asterisks,
    # with no possibility of four asterisks.
    # print('chi-squared:', chi2)
    # print('p-value:', p)
    return chi2, p


def format_pval(p):
    # APA (American Psychological Association) style, which shows three digits but omits the leading zero (.123).
    # P values less than 0.001 shown as "< .001". All P values less than 0.001 are summarized with three asterisks,
    # with no possibility of four asterisks.
    if 0.01 < p <= 0.05:
        return f"${str(round(p, 3)).strip('0')}^{{*}}$"
    elif 0.001 < p <= 0.01:
        return f"${str(round(p, 3)).strip('0')}^{{**}}$"
    elif 0.0 < p < 0.001:
        return "$<.001^{{***}}$"
    elif p == 0.0:
        return "-"
    else:
        return str(round(p, 3)).strip("0")




def export_wandb(projects, entity="gillesjacobs"):

    wandb_api = wandb.Api()
    for proj in projects:
        print(f"WandB.ai export download: {proj}")
        runs = wandb_api.runs(f"{entity}/{proj}")
        summary_list = []
        config_list = []
        name_list = []
        for run in runs:
            # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)
            # run.config is the input metrics.  We remove special values that start with _.
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )
            # run.name and run.id is the name/id of the run.
            name_list.append({"run_name": run.name, "run_id": run.id})
        summary_df = pd.DataFrame.from_records(summary_list)
        config_df = pd.DataFrame.from_records(config_list)
        name_df = pd.DataFrame.from_records(name_list)
        all_df = pd.concat([name_df, config_df, summary_df], axis=1)
        all_df.to_csv(
            hyperopt_dirp / f"{proj}-{pd.Timestamp.now()}.csv", index=False
        )  # write for recordkeeping


def load_preds(fp, return_target=False):
    # print(f'Loading preds from {dataset_fp}')
    df = pd.read_csv(fp)
    y_pred = df["pred"].to_numpy()
    if return_target:
        y_true = df["labels"].to_numpy()
        return y_pred, y_true
    else:
        return y_pred


def format_float(f, fmt="{:,.1f}", remove_trailing_zero=True, percent=True):

    if isinstance(f, float):
        if percent:
            f = 100.0 * f
        f_fmt = fmt.format(f)
        if remove_trailing_zero:
            f_fmt = f_fmt.lstrip("0")
        return f_fmt
    else:
        return f


def bold_extreme_values(data, max=-1, float_fmt=format_float):

    if data == float(max):
        data_str = (
            float_fmt.format(data) if isinstance(float_fmt, str) else float_fmt(data)
        )
        return "\\textbf{%s}" % data_str

    return data


def col_sort(c):
    if c.split("_")[-1] == "dev":
        return 0
    elif c.split("_")[-1] == "test":
        return 1
    elif "p-vs" in c:
        return 2


def format_metric(val, tag):
    if "{}" in tag:
        return tag.replace("}", f"{val}}}")
    else:
        return f"{tag} {val}"


def format_arch_names(arx):
    arx_rename = {
        "prosusai_finbert": "FinBERT-SST$_{Base}$ \citep{araci2019finbert}",
        "bert_base_cased": "BERT$_{Base}$ \citep{devlin-etal-2019-bert}",
        "bert_large_cased": "BERT$_{Large}$ \citep{devlin-etal-2019-bert}",
        "microsoft_deberta_base": "DeBERTa$_{Base}$ \citep{he2021deberta}",
        "roberta_base": "RoBERTa$_{Base}$ \citep{liu2019roberta}",
        "roberta_large": "RoBERTa$_{Large}$ \citep{liu2019roberta}",
    }

    return arx_rename[arx]

def load_dataset(fp_train, fp_dev, fp_test, args):
    with open(fp_train) as ftrain, open(fp_dev, "rt") as fdev, open(fp_test, "rt") as ftest:
        train_sentence_packs = json.load(ftrain)
        dev_sentence_packs = json.load(fdev)
        fulldevset_sentence_packs = train_sentence_packs + dev_sentence_packs # holdin
        test_sentence_packs = json.load(ftest) # holdout
    random.shuffle(fulldevset_sentence_packs)
    instances_train = load_data_instances(fulldevset_sentence_packs, args)
    instances_test = load_data_instances(test_sentence_packs, args)
    trainset = DataIterator(instances_train, args)
    testset = DataIterator(instances_test, args)

    return trainset, testset


def set_args(r):
    args = argparse.Namespace()
    for c, v in r.items():
        print(c, v)
        args.__setattr__(c, v)
    # some training params not in some runs -> add in manually
    args.__setattr__("patience_init", 8)
    args.__setattr__("patience", 16)
    print(vars(args))
    return args

if __name__ == "__main__":

    exp_name = "triplet-joinedsemeval" # senti is on sentivent-devproto data, joinedsemeval on joint Pontiki sets
    EXPORT_WANDB = False  # export overview from wandb API, else load from local storage, first run requires retrieval
    selection_metric = "best_joint_f1"

    hyperopt_dirp = Path(f"../../data/{exp_name}-results-hyperparameter-search")
    winpred_dirp = hyperopt_dirp / "winpreds/"
    winpred_dirp.mkdir(parents=True, exist_ok=True)
    # 0. Export all results from wandb first, can be skipped if data is collected
    # list hyperopt architecture projects to include here
    exp_projects = {
        "triplet-sentivent": [
            "roberta_base-triplet-sentivent",
            "roberta_large-triplet-sentivent",
            "bert_base_cased-triplet-sentivent",
            "bert_large_cased-triplet-sentivent",
            "prosusai_finbert-triplet-sentivent",
            "microsoft_deberta_base-triplet-sentivent",
        ],
        "triplet-joinedsemeval": [
            'roberta_large-triplet-joinedsemeval',
            'roberta_base-triplet-joinedsemeval',
        ],
    }
    projects = exp_projects[exp_name]

    dataset_dirs = {"triplet-sentivent": "sentivent-event-devproto", "triplet-joinedsemeval": "joinedsemeval"}
    dataset_dir = Path(f"../../data/{dataset_dirs[exp_name]}/")
    fp_train = dataset_dir / "train.json"
    fp_dev = dataset_dir / "dev.json"
    fp_test = dataset_dir / "test.json"

    if EXPORT_WANDB:
        export_wandb(projects)

    # A. Read all hyperparameter search results by architecture and select winning model.

    # A.1. Read in results from wandb.ai exports.
    df_win = pd.DataFrame()
    for arch_fp in hyperopt_dirp.glob("*.csv"):
        print(arch_fp)
        df = pd.read_csv(arch_fp)
        project_name = arch_fp.name.split("-202")[0]
        arch_name = project_name.replace(f"-{exp_name}", "")
        df["arch_name"] = arch_name
        df["model_name"] = "-".join(arch_name.split("-")[1:])
        # A.2. select winning model: highest devset macro-F1 that does not overfit on holdout testset (within 5% performance)
        winner = df.loc[df[selection_metric].idxmax()].copy()
        print(f"{arch_name} winner dev:")
        print(
            f"\t{selection_metric}:\t{winner[selection_metric]}"
        )
        df_win = df_win.append(winner, ignore_index=True)
        print(winner.best_epoch)
    df_win.to_csv("winner_dev0.tsv", sep="\t")
    # A.2 get best hyperparams + retrain on dev+train and test on holdout
    int_cols = ["batch_size", 'bert_feature_dim', 'best_epoch', 'class_num', 'epochs', 'nhops', 'max_sequence_len',
                'local_rank']
    df_win[int_cols] = df_win[int_cols].astype('Int64')
    df_win = df_win.sort_values(by=selection_metric, ascending=False)
    # best = df_win.iloc[0]
    df_win.to_csv("winner_dev.tsv", sep="\t")

    # B.1 RETRAIN WIN on HOLDIN + TEST & COLLECT PREDS ON HOLDOUT TESTSET
    # df_win = df_win.head(1) # testing
    n_runs = 5
    df_test = pd.DataFrame()
    test_win_score = 0
    for i, r in df_win.iterrows():
        res_fp = winpred_dirp / f"{exp_name}-{r.arch_name}-holdout-result-preds.tsv"
        if not res_fp.exists():
            try:
                # r["best_epoch"] = 1 # testing
                args = set_args(r)
                trainset, testset = load_dataset(fp_train, fp_dev, fp_test, args)
                all_res = []
                for i in range(n_runs): # train 5 times and average like in GTS paper
                    res = {"best_joint_f1": 0.0}
                    while res["best_joint_f1"] <= 0.0: # occassionally no convergence
                        print(f'\n===========================\n'
                              f'TRAINING {r.arch_name.upper()} {i+1}/{n_runs}'
                              f'\n===========================')
                        res = retrain_test()
                    all_res.append(res)
                df_res = pd.DataFrame(all_res)
                df_res.to_csv(res_fp, sep="\t", index=False)
            except Exception as e:
                print(f"FAILED TRAINING {r['arch_name']}: {e}")
                continue

        # C. COLLECT RESULT AND OUTPUT
        # C.1. Load results and take mean across 5 random init train+test
        print(f'{r.arch_name.upper()}: Load results and preds.')
        df_res = pd.read_csv(res_fp, sep="\t")
        test_mean = df_res.mean()
        # C.2 Collect best-of-best holdout predictions for manual eval.
        if test_mean["best_joint_f1"] > test_win_score:
            test_win_score = test_mean["best_joint_f1"]
            win_preds = df_res.iloc[df_res["best_joint_f1"].idxmax]["best_preds"]
            win_arx = r.arch_name

        test_row = pd.concat([r, test_mean.add_prefix('test_')])
        df_test = df_test.append(test_row, ignore_index=True)

    df_test = df_test.set_index('arch_name')
    # 'best' refers to best epoch in training, scores are still mean over 5 runs.
    rel_cols = [c for c in df_test.columns for s in ['precision', 'recall', 'f1', 'arch'] if s in c and 'best' in c]
    rel_cols_order = [
        'best_joint_precision', 'test_best_joint_precision', 'best_joint_recall', 'test_best_joint_recall',
        'best_joint_f1', 'test_best_joint_f1',
        'test_best_aspect_precision', 'test_best_aspect_recall', 'test_best_aspect_f1',
        'test_best_opinion_precision', 'test_best_opinion_recall', 'test_best_opinion_f1',
    ]
    rel_cols.sort(key=lambda x: rel_cols_order.index(x))
    df_test = df_test[rel_cols]
    print(rel_cols)
    print(df_test.head())
    print(df_test.columns)
    print(f'Win {win_arx} preds: {win_preds[0:64]}')

    # print("Computing Cochran's test across all predictions...") # this can take a while for a lot of runs
    # q, p_value = cochrans_q(test_target, *df_win["test_preds"].to_list())
    # print('\tCochran\'s Q: %.3f' % q)
    # print(f"\tp-value: {p_value}") # if p-value < 0,05 -> h0 that there is no difference in clf accuracies rejected

    # for i, r in df_win.iterrows():
    #     print(f"{r['arch_name']} pairwise t-test with best preds")
    #     test_pred = r["test_preds"]
    #     dev_pred = r["dev_preds"]
    #
    #     print("DEV CLF REPORT")
    #     print(classification_report(dev_target, dev_pred))
    #     print("TEST CLF REPORT")
    #     print(classification_report(test_target, test_pred))
    #
    #     print(f"{r['arch_name']} vs {best['arch_name']}")
    #     chi2, p = test_mcnemar(
    #         test_target,
    #         test_pred,
    #         test_pred_best,
    #     )
    #     df_win.loc[i, f"chi2-vs-{best['arch_name']}"] = chi2
    #     df_win.loc[i, f"p-vs-{best['arch_name']}"] = p

    # Format table
    df_table = df_test
    float_fmt = "{:,.1f}"

    df_table = df_table.sort_index()
    df_table.columns = df_table.columns.str.replace("_best", "")

    # find global and local arch max indices
    metric_cols = [c for c in df_table.columns if "p-vs" not in c]
    max_glob_ixs = [[df_table[c].idxmax(), c] for c in metric_cols]
    # max_loc_ixs = []
    # for arx, df_arx in df_table.groupby(level=0):
    #     max_loc_ixs.extend([[df_arx[c].idxmax(), c] for c in metric_cols])

    # FORMATTING (yeah that's a lot of code for table markup)
    # format the floats to percent
    df_table[metric_cols] = (df_table[metric_cols] * 100).round(decimals=1)
    # # format p-values
    # df_table[f"p-vs-{best['arch_name']}"] = df_table[f"p-vs-{best['arch_name']}"].apply(lambda x: format_pval(x))

    # format the score highlights
    for idx, c in max_glob_ixs:
        df_table.loc[idx, c] = format_metric(df_table.loc[idx, c], "\\textbf{}")
    # for idx, c in max_loc_ixs:
    #     df_table.loc[idx, c] = format_metric(df_table.loc[idx, c], "\\underline{}")

    # df_table = df_table.rename(columns={f"p-vs-{best['arch_name']}": "$p$"})
    df_table.columns = df_table.columns.str.replace("_", "-")  # remove underscores
    df_table = df_table.replace("_", "-", regex=True)
    df_table.index = [
        format_arch_names(m) for m in df_table.index
    ]

    aspect_opinion_cols = [c for c in df_table.columns for s in ['opinion', 'aspect'] if s in c]
    triplet_cols = [c for c in df_table.columns if 'joint' in c]
    df_triplet = df_table[triplet_cols]
    df_oa = df_table[aspect_opinion_cols]

    print(f"\n\section{{{exp_name} DEV+TEST results table}}\n")
    caption_triplet = f"Fine-grained token-level implicit sentiment results on development set and holdout test set for winning models after optimising hyperparameters for each architecture.  " \
                      f"Precision (P), recall (R), $F_1$-score ($F_1$) percentages are micro-averaged.)"
    print(
        post_process_latex(
            df_triplet.to_latex(
                escape=False,
                index='arch_name',
                caption=caption_triplet,
                label="tab:triplet-result-dev+test",
            ),
            len_header=8,
        )
    )

    caption_ao = "Fine-grained aspect and opinion term extraction results on holdout testset for winning models after optimising hyperparameters for each architecture.  Precision (P), recall (R), $F_1$-score ($F_1$) percentages are micro-averaged.)"
    print(
        post_process_latex(
            df_oa.to_latex(
                escape=False,
                index='arch_name',
                caption=caption_ao,
                label="tab:aspect-opinion-result-dev+test",
            ),
            len_header=8,
        )
    )