from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import random

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


class AverageMeter(object):
    """Computes and stores the average and current value"""
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



def compute_multilabel_metrics(output, target, threshold=0.3):
    with torch.no_grad():
        probs = output
        # Binarize predictions and target
        pred = (output > threshold).int()   # Apply threshold to predictions
        target = target.int()      # Ensure target is binary (multi-hot)

        # Hamming Loss
        hamming_loss = (pred != target).float().mean().item()  # Fraction of incorrect labels
        hamming_accuracy = 1 - hamming_loss  # Complement of loss

        # Subset Accuracy (Exact Match Ratio)
        subset_accuracy = (pred == target).all(dim=1).float().mean().item()

        # True Positives, False Positives, False Negatives
        tp = (pred & target).sum(dim=0).float()  # True positives per label
        fp = (pred & ~target).sum(dim=0).float()  # False positives per label
        fn = (~pred & target).sum(dim=0).float()  # False negatives per label

        # Precision, Recall, F1-Score per label
        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-8)     # Avoid division by zero
        f1 = 2 * precision * recall / (precision + recall + 1e-8)  # F1-Score

        # Average Precision, Recall, and F1-Score (Macro Averaging)
        avg_precision = precision.mean().item()
        avg_recall = recall.mean().item()
        avg_f1 = f1.mean().item()

        # Mean Average Precision (mAP)
        average_precisions = []  # Store AP for each class
        for i in range(output.size(1)):  # Loop over each label
            target_label = target[:, i].cpu().numpy()
            prob_label = probs[:, i].cpu().numpy()
            
            # Sort predictions and targets by probability in descending order
            sorted_indices = prob_label.argsort()[::-1]
            sorted_target = target_label[sorted_indices]
            
            # Compute Precision-Recall curve
            tp_cumsum = sorted_target.cumsum()
            fp_cumsum = (~sorted_target.astype(bool)).cumsum()
            
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            recalls = tp_cumsum / (tp_cumsum[-1] + 1e-8)  # Total positives in this class
            
            # Average Precision (AP) is the area under the Precision-Recall curve
            ap = (precisions[1:] * (recalls[1:] - recalls[:-1])).sum()
            average_precisions.append(ap)
        
        mean_ap = sum(average_precisions) / len(average_precisions)


    # Return metrics as a dictionary


    # Results
    return mean_ap, avg_recall, avg_f1, hamming_accuracy, subset_accuracy


def best_metrics(output,target):
    th = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    best_mean_ap = 0
    best_recall = 0
    best_f1 = 0
    best_hamming_accuracy = 0
    best_subset_accuracy = 0
    
    for i in th:
        mean_ap, avg_recall, avg_f1, hamming_accuracy, subset_accuracy = compute_multilabel_metrics(output, target, threshold=i)
        
        best_mean_ap = max(best_mean_ap,mean_ap)
        best_recall = max(best_recall,avg_recall)
        best_f1 = max(best_f1,avg_f1)
        best_hamming_accuracy = max(best_hamming_accuracy,hamming_accuracy)
        best_subset_accuracy = max(best_subset_accuracy,subset_accuracy)
    return best_mean_ap, best_recall, best_f1, best_hamming_accuracy, best_subset_accuracy


def compute_label_specific_thresholds(output, target, thresholds=np.linspace(0.1, 0.9, 9)):
    """
    Compute the best threshold for each label based on a specific metric (e.g., F1 score).
    """
    num_labels = output.size(1)  
    best_thresholds = [0.0] * num_labels  
    best_f1_scores = [0.0] * num_labels  

    for label_idx in range(num_labels):
        for threshold in thresholds:
            # Calculate metrics for the current threshold and label
            mean_ap, avg_recall, avg_f1, _, _ = compute_multilabel_metrics(
                output[:, [label_idx]], target[:, [label_idx]], threshold=threshold
            )
            # Update the best threshold for this label if the F1 score improves
            if avg_f1 > best_f1_scores[label_idx]:
                best_f1_scores[label_idx] = avg_f1
                best_thresholds[label_idx] = threshold

    return best_thresholds


def best_metrics_with_label_specific_thresholds(output, target):
    """
    Use label-specific thresholds to compute the best metrics.
    """
    # Compute the best threshold for each label
    label_specific_thresholds = compute_label_specific_thresholds(output, target)

    # Apply label-specific thresholds to compute metrics
    with torch.no_grad():
        predictions = torch.zeros_like(output, dtype=torch.int)  # Placeholder for predictions
        for i, threshold in enumerate(label_specific_thresholds):
            predictions[:, i] = (output[:, i] > threshold).int()  # Threshold each label

        # Compute metrics using the final label-specific predictions
        mean_ap, avg_recall, avg_f1, hamming_accuracy, subset_accuracy = compute_multilabel_metrics(
            predictions.float(), target.float()
        )

    return {
        "mean_ap": mean_ap,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "hamming_accuracy": hamming_accuracy,
        "subset_accuracy": subset_accuracy,
    }




def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr




def save_model(model, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

print("all worked --- no error")