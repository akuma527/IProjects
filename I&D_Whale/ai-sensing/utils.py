from torch.autograd import Variable
import torch
import numpy as np
from process.data_helper import *
import os


def save(list_or_dict, name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()


def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp


def dot_numpy(vector1, vector2, emb_size=512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1, 0)

    return np.dot(vector1, vector2)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def top_preds(preds, labels, n=5):
    correct = 0
    top5 = 0
    for i in range(len(labels)):
        classes = (-preds[i]).argsort()
        if classes[0] == labels[i]:
                correct += 1
        for c in classes[:n]:
            if c == labels[i]:
                top5 += 1
                break
    return correct / len(labels), top5 / len(labels)


def metric(prob, label, thres=0.0):
    shape = prob.shape
    prob_tmp = np.ones([shape[0], shape[1] + 1]) * thres
    prob_tmp[:, :shape[1]] = prob
    precision, top5 = top_n_np(prob_tmp, label)
    return precision, top5


def top_n_np(preds, labels):
    n = 5
    predicted = np.fliplr(preds.argsort(axis=1)[:, -n:])
    top5 = []

    re = 0
    for i in range(len(preds)):
        predicted_tmp = predicted[i]
        labels_tmp = labels[i]
        for n_ in range(5):
            re += np.sum(labels_tmp == predicted_tmp[n_]) / (n_ + 1.0)

    re = re / len(preds)
    for i in range(n):
        top5.append(np.sum(labels == predicted[:, i])/ (1.0*len(labels)))
    return re, top5


def make_train_dirs(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'checkpoint')):
        os.makedirs(os.path.join(out_dir, 'checkpoint'))
    if not os.path.exists(os.path.join(out_dir, 'train')):
        os.makedirs(os.path.join(out_dir, 'train'))
    if not os.path.exists(os.path.join(out_dir, 'backup')):
        os.makedirs(os.path.join(out_dir, 'backup'))


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min'%(hr,min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError


def f1_at_n(is_match, potential_matches, n):
    """
    Takes a boolean list denoting if the n-th entry of the predictions is an actual match
    and the number of potential matches, i.e. how many matches are at most possible and
    an integer n and computed the f1 score if one were to only consider the n most
    relevant matches
    :param is_match:
    :param potential_matches:
    :param n:
    :return:
    """
    if potential_matches == 0:
        return 0
    correct_prediction = float(sum(is_match[:n]))
    precision = correct_prediction / n
    recall = correct_prediction / potential_matches
    try:
        if (recall + precision) != 0.0:
            f1 = 2 * (recall * precision) / (recall + precision)
        else:
            f1 = 0
    except ZeroDivisionError:
        f1 = 0
    return f1



