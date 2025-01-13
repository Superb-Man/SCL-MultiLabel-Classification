import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Modifying the Supervised Contrastive Loss for Multi-Label Data
    source reference: https://github.com/HobbitLong/SupContrast
    Modified by : Kazi Istiak Uddin Toriqe
    

"""


"""
    Example formulation :
        # asumming number of labels = 5
        y1_label = [1, 0, 1, 0, 1]
        y2_label = [0, 1, 0, 1, 1]
        y3_label = [1, 0, 1, 1, 1]


        y1_label_INTER_y2_label = [0, 0, 0, 0, 1] = 1
        y1_label_UNION_y2_label = [1, 1, 1, 1, 1] = 4
        
        y1_label_INTER_y3_label = [1, 0, 1, 0, 1] = 3
        y1_label_UNION_y3_label = [1, 0, 1, 1, 1] = 4

        y2_label_INTER_y3_label = [0, 0, 0, 1, 1] = 2
        y2_label_UNION_y3_label = [1, 1, 1, 1, 1] = 4


        Jaccard Similarity_y1_y2 = 1/4 = 0.25
        Jaccard Similarity_y1_y3 = 3/4 = 0.75
        Jaccard Similarity_y2_y3 = 2/4 = 0.50

        Make set of positive pairs based on similarity threshold

        threshold = 0.5

        Positive pair of y1 = P(y1) = {y3}
        Positive pair of y2 = P(y2) = {y3}
        Positive pair of y3 = P(y3) = {y1, y2}

        Negative pair of y1 = N(y1) = {y2}
        Negative pair of y2 = N(y2) = {y1}
        Negative pair of y3 = N(y3) = {}


"""
class MultiSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, similarity_threshold=0.5, c_treshold=0.3):
        super(MultiSupConLoss, self).__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        self.c_treshold = c_treshold

    def forward(self, features, labels):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [batch_size, n_views, feature_dim].")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) 
        print("Contrast Feature: ", contrast_feature)


        mask = torch.zeros((batch_size, batch_size), dtype=torch.float32, device=device)

        intersection = torch.matmul(labels.float(), labels.T.float())

        label_sum = labels.sum(dim=1, keepdim=True).float() 
        union = label_sum + label_sum.T - intersection

        # Compute Jaccard similarity
        """
        Example:
            by matrix multiplication we get the intersection of labels

            labels = [ [1, 0, 1], 
                       [0, 1, 0], 
                       [1, 0, 1], 
                       [0, 1, 1] ]

            intersection = [ [2, 0, 2, 1], 
                             [0, 1, 0, 1], 
                             [2, 0, 2, 1], 
                             [1, 1, 1, 2] ]


            union = [ [2, 1, 2, 2], 
                      [1, 1, 1, 2], 
                      [2, 1, 2, 2], 
                      [2, 2, 2, 3] ]

            jacard_similarity = [  [2/2, 0/1, 2/2, 1/2],
                                    [0/1, 1/1, 0/1, 1/2],
                                    [2/2, 0/1, 2/2, 1/2],
                                    [1/2, 1/2, 1/2, 2/3] ]

            mask = [ [1, 0 ,1,1],
                     [0, 1, 0, 1],
                     [1, 0, 1, 1],
                     [1, 1, 1, 1] ]

            weights = [[3.3333, 0.0000, 3.3333, 1.1111, 3.3333, 0.0000, 3.3333, 1.1111],
                        [0.0000, 3.3333, 0.0000, 1.6667, 0.0000, 3.3333, 0.0000, 1.6667],
                        [3.3333, 0.0000, 3.3333, 1.1111, 3.3333, 0.0000, 3.3333, 1.1111],
                        [1.1111, 1.6667, 1.1111, 3.3333, 1.1111, 1.6667, 1.1111, 3.3333],
                        [3.3333, 0.0000, 3.3333, 1.1111, 3.3333, 0.0000, 3.3333, 1.1111],
                        [0.0000, 3.3333, 0.0000, 1.6667, 0.0000, 3.3333, 0.0000, 1.6667],
                        [3.3333, 0.0000, 3.3333, 1.1111, 3.3333, 0.0000, 3.3333, 1.1111],
                        [1.1111, 1.6667, 1.1111, 3.3333, 1.1111, 1.6667, 1.1111, 3.3333]]

            similaritis = contrast_feature * contrast_feature.T / temperature


            
        """
        jaccard_similarity = intersection / (union + 1e-8) 


        mask = (jaccard_similarity >= self.c_treshold).float()
        # print("Mask: ", mask)

        weights = jaccard_similarity / (self.c_treshold + 1e-8)
        weights = weights.repeat(contrast_count, contrast_count)


        dot_similarities = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T), self.temperature
        )  

        logits_max, _ = torch.max(dot_similarities, dim=1, keepdim=True)
        logits = dot_similarities - logits_max.detach()

        mask = mask.repeat(contrast_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)


        mean_log_prob_pos = (mask * log_prob * weights).sum(1) / (mask.sum(1) + 1e-8)
        loss = -(self.temperature * mean_log_prob_pos).mean()

        return loss
    
class MultiSupConLoss2(nn.Module):
    def __init__(self, temperature=0.07, similarity_threshold=0.5):
        super(MultiSupConLoss2, self).__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold

    def forward(self, features, labels):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` should have at least 3 dimensions [batch_size, n_views, feature_dim].")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Compute Jaccard similarity
        intersection = torch.matmul(labels, labels.T).float()
        union = labels.sum(dim=1, keepdim=True) + labels.sum(dim=1, keepdim=True).T - intersection
        similarity = intersection / (union + 1e-6)

        # Precomputing positive and negative masks
        positive_mask = (similarity >= self.similarity_threshold).float().to(device)
        negative_mask = (similarity < self.similarity_threshold).float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feature_dim]
        anchor_count = features.shape[1]

        dot_similarities = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature  # [N_views * B, N_views * B]

        logits_max, _ = torch.max(dot_similarities, dim=1, keepdim=True)
        logits = dot_similarities - logits_max.detach()

        # Mask self-contrast cases
        logits_mask = torch.ones_like(logits)
        idx = torch.arange(batch_size * anchor_count, device=device)
        logits_mask = logits_mask.scatter(1, idx.view(-1, 1), 0) 
        
        positive_mask = positive_mask.repeat(anchor_count, anchor_count) * logits_mask
        negative_mask = negative_mask.repeat(anchor_count, anchor_count) * logits_mask

        losses = []
        for i in range(batch_size):
            positive_indices = torch.where(positive_mask[i] > 0)[0]
            negative_indices = torch.where(negative_mask[i] > 0)[0]

            if len(positive_indices) == 0:
                continue

            pos_logits = logits[i, positive_indices]
            neg_logits = logits[i, negative_indices]

            # Compute loss for current sample
            sample_loss = -torch.log(torch.exp(pos_logits).sum() / (torch.exp(pos_logits).sum() + torch.exp(neg_logits).sum() + 1e-8))
            losses.append(sample_loss)

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True, device=device)

        return loss
    

class ClassWiseSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ClassWiseSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [batch_size, n_views, feature_dim].")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feature_dim]

        losses = []
        for class_idx in range(labels.shape[1]):  # Iterate over classes
            class_mask = labels[:, class_idx]  # [batch_size]
            positive_indices = torch.where(class_mask == 1)[0]

            if len(positive_indices) < 2:
                continue 

            # Class-specific features
            class_features = contrast_feature[positive_indices.repeat_interleave(contrast_count)]
            dot_similarities = torch.matmul(class_features, class_features.T) / self.temperature  # Cosine similarity

            logits_max, _ = torch.max(dot_similarities, dim=1, keepdim=True)
            logits = dot_similarities - logits_max.detach()

            logits_mask = torch.ones_like(logits)
            idx = torch.arange(len(class_features), device=device)
            logits_mask.scatter_(1, idx.view(-1, 1), 0)

            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

            mean_log_prob_pos = log_prob.mean()
            losses.append(-mean_log_prob_pos)

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True, device=device)
        return loss


class LabelMaskingSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(LabelMaskingSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [batch_size, n_views, feature_dim].")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feature_dim]

        # Compute label similarity mask
        label_sim = torch.matmul(labels.float(), labels.T.float())  # [batch_size, batch_size]
        positive_mask = (label_sim > 0).float().to(device)
        positive_mask = positive_mask.repeat(contrast_count, contrast_count)

        # Compute logits
        dot_similarities = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature  # [batch_size * n_views, batch_size * n_views]

        logits_max, _ = torch.max(dot_similarities, dim=1, keepdim=True)
        logits = dot_similarities - logits_max.detach()

        logits_mask = torch.ones_like(logits)
        idx = torch.arange(batch_size * contrast_count, device=device)
        logits_mask.scatter_(1, idx.view(-1, 1), 0)  # Mask self-comparisons

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)
        loss = -(mean_log_prob_pos).mean()

        return loss



class MultiLabelContrastiveLoss(nn.Module):
    def __init__(self, alpha=1.0, temperature=0.07, reduction='mean', beta=1.0):
        super(MultiLabelContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        self.beta = beta

    def forward(self, features, labels):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` should have at least 3 dimensions [batch_size, n_views, feature_dim].")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feature_dim]

        # Compute label similarities using Jaccard similarity
        intersection = torch.matmul(labels, labels.T).float()
        union = labels.sum(dim=1, keepdim=True) + labels.sum(dim=1, keepdim=True).T - intersection
        similarity = intersection / (union + 1e-8)

        # Weighting function f(y_i, y_j)
        weight_function = (intersection / (union + 1e-8)) ** self.beta

        expanded_similarity = similarity.repeat_interleave(contrast_count, dim=0).repeat_interleave(contrast_count, dim=1)
        expanded_weights = weight_function.repeat_interleave(contrast_count, dim=0).repeat_interleave(contrast_count, dim=1)

        # Create positive and negative masks
        positive_mask = (expanded_similarity > 0).float().to(device)  # Positive pairs
        negative_mask = 1 - positive_mask  # Negative pairs

        dot_similarities = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature

        # Mask self-contrast
        logits_max, _ = torch.max(dot_similarities, dim=1, keepdim=True)
        logits = dot_similarities - logits_max.detach()
        logits_mask = torch.ones_like(logits)
        idx = torch.arange(batch_size * contrast_count, device=device)
        logits_mask.scatter_(1, idx.view(-1, 1), 0)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute positive and negative contributions with weights
        positive_log_prob = (positive_mask * expanded_weights * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)
        negative_log_prob = (negative_mask * log_prob).sum(1) / (negative_mask.sum(1) + 1e-8)

        # Combine with weighting factor
        loss = -self.alpha * positive_log_prob - (1 - self.alpha) * negative_log_prob

        regularization = self._compute_regularization(contrast_feature)
        loss += regularization

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _compute_regularization(self, embeddings):
        reg_loss = torch.mean(torch.norm(embeddings, dim=1))
        return reg_loss

def main():
    torch.manual_seed(0)
    labels = torch.tensor([
        [1, 0, 1],  # Sample 0: belongs to classes 0 and 2
        [0, 1, 0],  # Sample 1: belongs to class 1
        [1, 0, 1],  # Sample 2: belongs to classes 0 and 2 (similar to Sample 0)
        [0, 1, 1],  # Sample 3: belongs to classes 1 and 2
    ])

    # Dummy features: [batch_size, n_views, feature_dim]
    # Assume 2 views for each sample, feature_dim = 4
    features = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.1, -0.1]], 
        [[0.0, 1.0, 0.0, 0.0], [0.1, 0.9, -0.1, 0.0]],  
        [[0.9, 0.1, 0.0, 0.0], [0.8, 0.2, -0.2, 0.1]], 
        [[0.0, 0.9, 0.1, 0.0], [0.2, 0.8, 0.2, -0.1]],
    ])
    multi_label_loss = MultiSupConLoss()
    multi_label_loss2 = MultiSupConLoss2()
    # multi_label_loss3 = MultiSupConLoss3()
    # multi_label_loss4 = MultiLabelSupConLoss()
    multi_label_loss5 = ClassWiseSupConLoss()
    multi_label_loss6 = LabelMaskingSupConLoss()
    multi_label_loss7 = MultiLabelContrastiveLoss()


    multi_label_loss = multi_label_loss(features, labels)
    multi_label_loss2 = multi_label_loss2(features, labels)
    # multi_label_loss3 = multi_label_loss3(features, labels)
    # multi_label_loss4 = multi_label_loss4(features, labels)
    multi_label_loss5 = multi_label_loss5(features, labels)
    multi_label_loss6 = multi_label_loss6(features, labels)
    multi_label_loss7 = multi_label_loss7(features, labels)


    
    print("Computed Losses:")
    print(f"Multi-Label Supervised Contrastive Loss: {multi_label_loss.item():.6f}")
    print(f"Multi-Label Supervised Contrastive Loss: {multi_label_loss2.item():.6f}")
    # print(f"Multi-Label Supervised Contrastive Loss: {multi_label_loss3.item():.6f}")
    # print(f"Multi-Label Supervised Contrastive Loss: {multi_label_loss4.item():.6f}")
    print(f"Class-Wise Supervised Contrastive Loss: {multi_label_loss5.item():.6f}")
    print(f"Label-Masking Supervised Contrastive Loss: {multi_label_loss6.item():.6f}")
    print(f"Multi-Label Contrastive Loss: {multi_label_loss7.item():.6f}")

if __name__ == "__main__":
    main()


