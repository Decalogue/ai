# -*- coding: utf-8 -*-
""" ai.metric """
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.metrics.distance import edit_distance


class AverageMeter(object):
    """ Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def corpus_ed(references, hypotheses):
    res = 0.
    for gt_group, pred in zip(references, hypotheses):
        norm_ed = 0
        for gt in gt_group:
            if len(gt) == 0:
                norm_ed += 1
            else:
                norm_ed += edit_distance(gt, pred) / len(gt)
        res += norm_ed
    res /= len(references)
    return res


def corpus_match(references, hypotheses):
    res = 0
    for gt_group, pred in zip(references, hypotheses):
        match = 0
        for gt in gt_group:
            if len(gt) != 0 and gt == pred:
                match += 1
        res += match
    res /= len(references)
    return res


def corpus_meteor(references, hypotheses):
    """ The original input format of Meteor metric is different form BLEU series.
        In this function, we change the format of BLEU to fit Meteor.
    """
    def to_str(values):
        return [str(val) for val in values]

    Meteor = 0.0
    for gt_group, pred in zip(references, hypotheses):
        gt = [' '.join(to_str(val)) for val in gt_group]
        pred = ' '.join(to_str(pred))
        Meteor += meteor_score(gt, pred)
    return Meteor / (len(references))


def translation_scores(references, hypotheses):
    """
        :param references: references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...]
        :param hypotheses: hypotheses = [hyp1, hyp2, ...]
    """
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    meteor = corpus_meteor(references, hypotheses)
    ed = corpus_ed(references, hypotheses)
    match = corpus_match(references, hypotheses)

    bleu1 = round(bleu1, 4)
    bleu2 = round(bleu2, 4)
    bleu3 = round(bleu3, 4)
    bleu4 = round(bleu4, 4)
    meteor = round(meteor, 4)
    ed = round(ed, 4)
    match = round(match, 4)

    return (bleu1, bleu2, bleu3, bleu4, meteor, ed, match)