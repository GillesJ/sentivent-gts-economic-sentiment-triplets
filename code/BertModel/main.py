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

from pathlib import Path

def train(args):

    # load dataset
    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
    random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))
    instances_train = load_data_instances(train_sentence_packs, args)
    instances_dev = load_data_instances(dev_sentence_packs, args)
    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model = MultiInferBert(args).to(args.device)

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': args.learning_rate},
        {'params': model.cls_linear.parameters()}
    ], lr=args.learning_rate)

    best_joint_f1 = 0
    best_joint_epoch = 0
    for i in range(args.epochs):
        print('Epoch:{}'.format(i))
        for j in trange(trainset.batch_count):
            _, tokens, lengths, masks, _, _, aspect_tags, tags = trainset.get_batch(j)
            preds = model(tokens, masks)

            preds_flatten = preds.reshape([-1, preds.shape[3]])
            tags_flatten = tags.reshape([-1])
            loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)

        if joint_f1 > best_joint_f1:
            model_dir = Path(args.model_dir) / args.trained_model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / 'dev_best.pt'
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
            best_joint_precision = joint_precision
            best_joint_recall = joint_recall
    print('best epoch: {}\tbest dev {} f1: {:.5f} p: {:.5f} r: {:.5f}\n\n'.format(best_joint_epoch,
                                                                                  args.task,
                                                                                  best_joint_f1,
                                                                                  best_joint_precision,
                                                                                  best_joint_recall))
    metadata = {
        "dev_best_joint_f1": best_joint_f1,
        "dev_best_joint_precision": best_joint_precision,
        "dev_best_joint_recall": best_joint_recall,
        "dev_best_joint_epoch": best_joint_epoch,
        "train_config": vars(args),
        }
    with open(Path(args.model_dir) / args.trained_model_name / 'metadata.json', "wt") as meta_out:
        json.dump(metadata, meta_out, indent=2)



def eval(model, dataset, args):
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
    return precision, recall, f1


def test(args):
    print("Evaluation on testset:")
    model_path = Path(args.model_dir) / args.trained_model_name / 'dev_best.pt'
    model = torch.load(model_path).to(args.device)
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    instances = load_data_instances(sentence_packs, args)
    testset = DataIterator(instances, args)
    eval(model, testset, args)
    # write testset for manual inspection
    testset_with_pred = [x.sentence_pack for x in testset.instances]
    with open(Path(args.model_dir) / args.trained_model_name / "test_preds.json", "wt") as pred_out:
        json.dump(testset_with_pred, pred_out, indent=2)

if __name__ == '__main__':
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
    parser.add_argument('--max_sequence_len', type=int, default=100,
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
    parser.add_argument('--trained_model_name', nargs='?', type=str,
                        default='', help='Name of trained model dir to save in/test from.')

    args = parser.parse_args()

    if not args.trained_model_name: # set unique model run id
        args.trained_model_name = f'{args.dataset}-{args.task}-{args.bert_model_path.replace("/", "_").replace("-","_")}' \
                     f'-nhops_{args.nhops}-sq_{args.max_sequence_len}_bs_{args.batch_size}_ep_{args.epochs}' \
                     f'_lr_{args.learning_rate}'

    if args.task == 'triplet':
        args.class_num = 6

    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)
