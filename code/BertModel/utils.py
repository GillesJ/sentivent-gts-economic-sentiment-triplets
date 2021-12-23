import multiprocessing
import pickle
import numpy as np
import sklearn

id2sentiment = {3: 'negative', 4: 'neutral', 5: 'positive'}


def get_human_readable(tuples, tokens):
    # generate human readable preds in format [al, ar, pl, pr, sentiment]
    insts_hr = []
    triplet_like_json = {}
    for (al, ar, pl, pr, sen) in tuples: # TODO change back to predicted_tuples
        triplet_like_json.setdefault((al, ar, sen), []).append((pl, pr))
    for (al, ar, sen), opinion_spans in triplet_like_json.items():
        asp_labels = ["O"] * len(tokens)
        asp_labels[al] = "B"
        asp_labels[al+1:ar] = ["I"] * (ar - al)

        triplet_hr = {
            "sentiment": id2sentiment[sen],
            "aspect_tokens": tokens[al:ar+1],
            "aspect_tags": " ".join(f'{t}\\{l}' for t, l in zip(tokens, asp_labels)),
        }

        opinions = []
        opi_labels_by_as = ["O"] * len(tokens)
        opi_toks_by_as = []

        for (pl, pr) in opinion_spans:
            opi_labels = ["O"] * len(tokens)
            opi_toks = tokens[pl:pr+1]
            opi_toks_by_as.append(opi_toks)
            opi_labels[pl] = "B"
            opi_labels[pl+1:pr] = ["I"] * (pr - pl)
            opi_labels_by_as[pl] = "B"
            opi_labels_by_as[pl+1:pr] = ["I"] * (pr - pl)
            opinions.append(
                {
                    "opinion_tokens": opi_toks,
                    "opinion_tags": " ".join(f'{t}\\{l}' for t, l in zip(tokens, opi_labels)),
                    "tuple_enc": (al, ar, pl, pr, sen)
                }
            )
        triplet_hr["opinions"] = opinions
        triplet_hr["opinion_tags_join"] = opi_toks_by_as
        triplet_hr["opinion_tokens_join"] = opi_toks_by_as
        triplet_hr["tuple_enc_join"] = (al, ar, sen, opinion_spans)
        insts_hr.append(triplet_hr)

    return insts_hr

def get_aspects(tags, length, ignore_index=-1):
    spans = []
    start = -1
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 1:
            if start == -1:
                start = i
        elif tags[i][i] != 1:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length-1])
    return spans


def get_opinions(tags, length, ignore_index=-1):
    spans = []
    start = -1
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 2:
            if start == -1:
                start = i
        elif tags[i][i] != 2:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length-1])
    return spans


class Metric():
    def __init__(self, args, predictions, goldens, bert_lengths, sen_lengths, tokens_ranges, ignore_index=-1):
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = -1
        self.data_num = len(self.predictions)

    def get_spans(self, tags, length, token_range, type):
        '''
        tags: encoded bert_tokens
        length: orginal sentence token length
        token_range: list from sentence_tok->bert_tok
        type: encoded label of interest
        '''
        spans = []
        start = -1
        for i in range(length): # for every token in orig sentence
            l, r = token_range[i] # get left-right-token bind
            if tags[l][l] == self.ignore_index: # if it is neg tag
                continue
            elif tags[l][l] == type: # if it is label
                if start == -1: # and start has not happened
                    start = i # label starts on this token
            elif tags[l][l] != type: # if it is other label
                if start != -1: # if start for our type has not happened, other type skip
                    spans.append([start, i - 1]) # get end label
                    start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def find_pair(self, tags, aspect_spans, opinion_spans, token_ranges):
        pairs = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 4
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                if tag_num[3] == 0: continue
                sentiment = -1
                pairs.append([al, ar, pl, pr, sentiment])
        return pairs

    def find_triplet(self, tags, aspect_spans, opinion_spans, token_ranges):
        triplets = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 6
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                        # if tags[i][j] != -1:
                        #     tag_num[int(tags[i][j])] += 1
                        # if tags[j][i] != -1:
                        #     tag_num[int(tags[j][i])] += 1
                if sum(tag_num[3:]) == 0: continue
                sentiment = -1
                if tag_num[5] >= tag_num[4] and tag_num[5] >= tag_num[3]:
                    sentiment = 5
                elif tag_num[4] >= tag_num[3] and tag_num[4] >= tag_num[5]:
                    sentiment = 4
                elif tag_num[3] >= tag_num[5] and tag_num[3] >= tag_num[4]:
                    sentiment = 3
                if sentiment == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    input()
                triplets.append([al, ar, pl, pr, sentiment])
        return triplets

    def score_aspect(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            for spans in golden_aspect_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_aspect_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            for spans in predicted_aspect_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_opinion(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_opinion_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            for spans in golden_opinion_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_opinion_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            for spans in predicted_opinion_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_uniontags(self, dataset):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            inst = dataset.instances[i]
            golden_aspect_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            golden_opinion_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            if self.args.task == 'pair':
                golden_tuples = self.find_pair(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                golden_tuples = self.find_triplet(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))
            inst.golden_enc = golden_tuples

            predicted_aspect_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            predicted_opinion_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            if self.args.task == 'pair':
                predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                predicted_tuples = self.find_triplet(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))
            inst.sentence_pack["gold_enc"] = get_human_readable(golden_tuples, inst.tokens)
            inst.sentence_pack["pred_enc"] = get_human_readable(predicted_tuples, inst.tokens)

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1