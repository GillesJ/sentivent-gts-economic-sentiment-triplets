#!/usr/bin/env python3
'''
Explain purpose of script here

hyperopt.py in GTS
7/5/21 Copyright (c) Gilles Jacobs
'''
#coding utf-8

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

def train_wandb():
    # Initialize a new wandb run
    with wandb.init(config=args) as run:
        train()
        run.finish() # Sync wandb

def train():
    global trainset
    if not os.path.exists(wandb.config.model_dir):
        os.makedirs(wandb.config.model_dir)

    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    model = MultiInferBert(wandb.config).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': wandb.config.learning_rate},
        {'params': model.cls_linear.parameters()}
    ], lr=wandb.config.learning_rate)

    model, optimizer, trainset = accelerator.prepare(model, optimizer, trainset)

    patience = 0
    init_patience = 0
    best_joint_f1 = -1
    best_joint_epoch = -1
    for i in range(wandb.config.epochs):
        print(f'Epoch:{i} (patience={patience}/{wandb.config.patience})')
        for j in trange(trainset.batch_count):
            _, tokens, lengths, masks, _, _, aspect_tags, tags = trainset.get_batch(j)
            preds = model(tokens, masks)

            preds_flatten = preds.reshape([-1, preds.shape[3]])
            tags_flatten = tags.reshape([-1])
            loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        joint_precision, joint_recall, joint_f1 = eval(model, devset)
        wandb.log({"joint_precision": joint_precision, "joint_recall": joint_recall, "joint_f1": joint_f1})

        if joint_f1 > best_joint_f1:
            patience = 0
            model_dir = Path(wandb.config.model_dir) / wandb.config.trained_model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / 'dev_best.pt'
            # torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
            best_joint_precision = joint_precision
            best_joint_recall = joint_recall
            wandb.log({"best_epoch": best_joint_epoch,
                       "best_joint_f1": best_joint_f1,
                       "best_joint_precision": best_joint_precision,
                       "best_joint_recall": best_joint_recall,
                       })
        else: # no improvement so increment patience
            patience += 1

        if patience == wandb.config.patience:
            print(f"Patience of {wandb.config.patience} exceeded, stopping run.")
            break

        if i == wandb.config.patience_init and best_joint_f1 <= 0:
            print(f"Initial patience of {wandb.config.patience_init} exceeded, stopping run.")
            break
    print('best epoch: {}\tbest dev {} f1: {:.5f} p: {:.5f} r: {:.5f}\n\n'.format(best_joint_epoch,
                                                                                  wandb.config.task,
                                                                                  best_joint_f1,
                                                                                  best_joint_precision,
                                                                                  best_joint_recall))

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

        metric = utils.Metric(wandb.config, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        precision, recall, f1 = metric.score_uniontags(dataset)
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()

        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))
        print(wandb.config.task + '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1


def test():
    print("Evaluation on testset:")
    model_path = Path(args.model_dir) / args.trained_model_name / 'dev_best.pt'
    model = torch.load(model_path).to(args.device)
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    instances = load_data_instances(sentence_packs, args)
    testset = DataIterator(instances, args)
    eval(model, testset)
    # write testset for manual inspection
    testset_with_pred = [x.sentence_pack for x in testset.instances]
    with open(Path(args.model_dir) / args.trained_model_name / "test_preds.json", "wt") as pred_out:
        json.dump(testset_with_pred, pred_out, indent=2)

if __name__ == '__main__':
    global trainset
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="pair", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                    help='option: train, test')
    parser.add_argument('--dataset', type=str, default="res14",
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=64,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')
    parser.add_argument('--bert_model_path', type=str,
                        default="pretrained/bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--nhops', type=int, default=1,
                        help='inference times')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=4,
                        help='label number')
    parser.add_argument('--patience', type=int, default=8,
                    help='amount of epochs with no improvement in target metric.')
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    parser.add_argument('--patience_init', type=int, default=8,
                help='amount of epochs from start with no improvement in target metric.')

    args = parser.parse_args()
    args.trained_model_name = f'{args.bert_model_path.replace("/", "_").replace("-","_")}-{args.task}-{args.dataset.split("-")[0]}'.lower()

    sweep_config = {
        "name": args.trained_model_name,
        "method": "bayes",
        "metric": {"name": "best_joint_f1", "goal": "maximize"},
        "parameters": { # SET for each model arch
            "learning_rate": {"min": 5e-5, "max": 1e-4},
            # "batch_size": {'values': [16, 32]},
            "nhops":  {'values': [0, 1, 2, 3]},
            },
        "early_terminate": {"type": "hyperband", "min_iter": 4,},
    }
    sweep_id = wandb.sweep(sweep_config, project=args.trained_model_name)

    # load dataset
    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
    random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))
    instances_train = load_data_instances(train_sentence_packs, args)
    instances_dev = load_data_instances(dev_sentence_packs, args)
    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)

    if args.task == 'triplet':
        args.class_num = 6

    if args.mode == 'train': # start sweep training
        wandb.agent(sweep_id, train_wandb)
        # test()
    else:
        test()