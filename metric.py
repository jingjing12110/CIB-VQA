# @File :metric.py
# @Time :2021/5/7
# @Desc :
import torch
import itertools


# ****************************************************
# CS(k)
# ****************************************************
def batch_accuracy(predicted, ground_true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = ground_true.gather(dim=1, index=predicted_index)
    return (agreeing * 0.3).clamp(max=1)


def calc_consistency(que_acc, k_val):
    """Implementation of Consensus Score (CS) formula from Shah et al. (2019):
       https://arxiv.org/abs/1902.05660
    :param que_acc: List of accuracies for a question grouping
    :param k_val: k from CS definition
    :return:
    """
    total_que_score = 0.
    n_c_k = 0.
    for Q_prime in itertools.combinations(que_acc, k_val):
        n_c_k += 1.
        total_que_score += float(all(score > 0. for score in Q_prime))
    return total_que_score / n_c_k


# ****************************************************
# #flips
# ****************************************************


