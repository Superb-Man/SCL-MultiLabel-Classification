import torch
import torch.nn as nn

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
    
class MultiLabelSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, similarity_threshold=0.5):
        super(MultiLabelSupConLoss, self).__init__()
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
        logits_mask = logits_mask.scatter(1, idx.view(-1, 1), 0)  # Set diagonal (self-similarity) to 0

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

        # Average loss across batch
        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True, device=device)

        return loss






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
    multi_label_loss2 = MultiLabelSupConLoss()


    multi_label_loss = multi_label_loss(features, labels)
    multi_label_loss2 = multi_label_loss2(features, labels)

    # loss = LabelWeightedConLoss()
    # loss = loss(features, labels)

    
    print("Computed Losses:")
    print(f"Multi-Label Supervised Contrastive Loss: {multi_label_loss.item():.6f}")
    print(f"Multi-Label Supervised Contrastive Loss: {multi_label_loss2.item():.6f}")
    # print(f"Label Weighted Contrastive Loss: {loss.item():.6f}")

if __name__ == "__main__":
    main()


