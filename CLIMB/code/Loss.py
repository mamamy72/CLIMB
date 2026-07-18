import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp


''' Loss Class 1: Reference Atlas Consistency Loss  '''

''' Loss 1 Parcellation similarity loss '''

class Parcellation_Similarity_Loss(nn.Module):

    def __init__(self, group_parcellation_labels):

        """

            Function: Use cross-entropy loss to learn group-level commonality and incorporate atlas prior information.

            Input: 1) group_parcellation_labels: Atlas prior

            Output: 1) loss: cross_entropy

        """

        super().__init__()

        self.atlas_labels = group_parcellation_labels

    def forward(self, assignment_output):

        return F.cross_entropy(assignment_output, self.atlas_labels, reduction='mean')

''' Loss 2 Feature consistency loss '''


class Feature_Consistency_Loss(nn.Module):

    def __init__(
            self,
            group_parcellation_labels,
            group_parcellation_features,
            individual_features_dict,
            age_list,
            eps=1e-8,
            threshold=1):

        super().__init__()

        """

            Function: Compute the Pearson correlation coefficients between parcels of the individual and the atlas, and use cosine similarity loss to learn feature consistency.

            Input: 1) group_parcellation_labels: Atlas prior
                   2) group_parcellation_features: Atlas features
                   3) individual_features_dict: Dictionary of individual features

            Output: 1) loss: cos loss

        """

        self.eps = eps
        self.threshold = threshold
        self.age_list = age_list
        self.individual_features_dict = individual_features_dict

        one_hot = F.one_hot(group_parcellation_labels, num_classes=-1).float()

        mass = one_hot.sum(dim=0)

        group_mean = (one_hot.T @ group_parcellation_features) / (mass.unsqueeze(1) + eps)

        group_mean = group_mean - group_mean.mean(dim=1, keepdim=True)

        self.group_mean = group_mean

        self.weight = {}

        for target_age in self.age_list:

            for source_age in self.age_list:

                if target_age != source_age:
                    age_weight = np.exp(-abs(target_age - source_age) / 800)

                    self.weight[(str(target_age), str(source_age))] = age_weight

    def compute_mean_feature(self, assignment_output, features):

        prob = F.softmax(assignment_output, dim=1)

        mass = prob.sum(dim=0)

        mean_feature = (prob.T @ features) / (mass.unsqueeze(1) + self.eps)

        mean_feature = (mean_feature - mean_feature.mean(dim=1, keepdim=True))

        return mean_feature

    def forward(self, assignment_outputs_dict):

        mean_feature_dict = {}

        for age in self.age_list:
            mean_feature_dict[str(age)] = self.compute_mean_feature(assignment_outputs_dict[str(age)],
                                                                    self.individual_features_dict[str(age)])

        group_loss = 0

        for age in self.age_list:
            feature = mean_feature_dict[str(age)]

            corr = F.cosine_similarity(feature, self.group_mean, dim=1)

            penalty = torch.pow(F.relu(self.threshold - corr), 2)

            group_loss += penalty.mean()

        # group_loss/=len(self.age_list)

        num_ages = len(self.age_list)

        if num_ages <= 1:
            return group_loss

        long_loss = 0
        pair_num = 0

        for target_age in self.age_list:

            target_feature = mean_feature_dict[str(target_age)]

            for source_age in self.age_list:

                if target_age != source_age:
                    source_feature = mean_feature_dict[str(source_age)].detach()

                    corr = F.cosine_similarity(target_feature, source_feature, dim=1)

                    pair_loss = torch.pow(F.relu(self.threshold - corr), 2)

                    pair_loss = (pair_loss * self.weight[(str(target_age), str(source_age))])

                    long_loss += pair_loss.mean()

                    pair_num += 1

        # long_loss/=pair_num

        total_loss = group_loss + 0.5 * long_loss

        return total_loss



''' Loss Class 2: Parcellation Homogeneity Loss '''

''' Loss 3 Dynamic spatial homogeneity loss '''

class Dynamic_Spatial_Homogeneity_Loss(nn.Module):

    def __init__(self, edge_index, individual_parcellation_features, pos, init_tau=2.0, min_tau=0.01, max_tau=5.0,decrease_factor=0.9, increase_factor=1.1, eps=1e-8, sigma=10.0, parcel_num=180, smooth_weight=0.25):

        """

            This is a novel parcellation homogeneity optimization loss. It primarily addresses the
            issue of fragmented parcels in homogeneity optimization caused by high data noise and strong connection,
            optimizing parcellation homogeneity while maximizing parcel contiguity.

            The innovative design to solve this problem stems from two key points:

            1. Feature Aggregation via 1-Ring Convolution:
               Given that high noise in certain vertex regions can lead to misallocation, the similarity
               calculation between a vertex and a cluster incorporates the features of surrounding vertices.
               We first apply a 1-ring neighborhood convolution to aggregate the features of adjacent vertices,
               which are then combined with the target vertex via weighted fusion. This significantly reduces
               the likelihood of vertex misclassification.

            2. Dynamic Spatial Weighting Mechanism:
               When calculating the similarity between a vertex and a cluster, spatial coordinates are introduced
               to constrain and prevent the formation of fragmented parcels. Notably, this spatial weight is
               initially set to a large value to ensure a vertex is not assigned to a distant cluster. However,
               an excessively large weight tends to force parcels into circular shapes, which violates
               neurobiological priors. Therefore, we continuously decay this weight during the iterative
               optimization process, provided that the decay does not lead to fragmentation. If fragmentation
               does occur, the spatial weight for that specific parcel is dynamically increased to prevent
               further deterioration.

            Finally, in addition to resolving the parcel fragmentation problem, we introduce a novel optimization
            objective: alongside optimizing intra-parcel homogeneity, we also optimize inter-parcel heterogeneity
            between adjacent parcels. This enhances the alignment between the parcellation boundaries and the
            underlying true functional boundaries. This performance can be evaluated using the Silhouette Coefficient
            or a distance-controlled boundary coefficient.


            Input: 1) edge_index: Records the adjacency relationships between vertices
                   2) individual_parcellation_features: Individual feature
                   3) pos: Spatial coordinates

            Output: 1) loss: homo loss

        """

        super().__init__()

        self.min_tau = min_tau
        self.max_tau = max_tau
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor
        self.eps = eps
        self.edge_index = edge_index
        self.individual_parcellation_features = individual_parcellation_features
        self.Pos = pos
        self.sigma = sigma
        self.smooth_weight=smooth_weight

        self.tau = torch.full((parcel_num,), init_tau, dtype=torch.float32, device='cuda:0')

        self.adj_norm_matrix = self._adj_norm(edge_index)

    @torch.no_grad()
    def _row_center_l2norm(self, X):

        mean = X.mean(dim=1, keepdim=True)
        Xc = X - mean
        l2 = torch.norm(Xc, p=2, dim=1, keepdim=True).clamp_min(self.eps)
        return Xc / l2

    def _adj_norm(self,neighbor_edge_index):

        N = self.individual_parcellation_features.shape[0]
        device = neighbor_edge_index.device

        dst = neighbor_edge_index[1]
        src = neighbor_edge_index[0]

        deg = torch.zeros(N, device=device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg.clamp_min(1)

        val = 1.0 / deg[dst]
        adj_norm = torch.sparse_coo_tensor(torch.stack([dst, src], dim=0), val, (N, N)).coalesce()

        return adj_norm

    def forward(self, P):

        P = F.softmax(P, dim=1)
        N, K = P.shape
        eps = self.eps

        """ Computes the weighted smoothed features for each vertex by incorporating its 1-ring neighborhood features. """
        Vertex_feature = self.smooth_weight * torch.sparse.mm(self.adj_norm_matrix, self.individual_parcellation_features) + (1-self.smooth_weight) * self.individual_parcellation_features
        # Vertex_feature = self.individual_parcellation_features
        Vertex_feature = self._row_center_l2norm(Vertex_feature)  # (N, d)

        """ Computes the intra-parcel homogeneity """
        B = Vertex_feature.T @ P
        mass = P.sum(dim=0)
        numerator = B.T @ B
        denom = (mass.unsqueeze(1) * mass.unsqueeze(0)).clamp_min(eps)
        cluster_sim = numerator / denom
        Feature_intra_homo = torch.diagonal(cluster_sim)

        """ Computes the heterogeneity between adjacent parcels """
        predict_label=torch.argmax(P, dim=1)
        one_hot_tensor = F.one_hot(predict_label, num_classes=180).float().detach()
        A = torch.sparse.mm(torch.sparse.mm(one_hot_tensor.T, self.adj_norm_matrix),one_hot_tensor)
        adj_mask = (A > 1e-6) & (~torch.eye(P.shape[-1], dtype=torch.bool).to('cuda:0'))
        if adj_mask.any():
            loss_hete = cluster_sim[adj_mask].mean()
        else:
            loss_hete = torch.tensor(0.0)

        """ Computes the intra-parcel spatial homogeneity """
        pos_mean = (P.T @ self.Pos) / (mass.unsqueeze(1) + self.eps)
        dist2 = (self.Pos ** 2).sum(dim=1, keepdim=True) + (pos_mean ** 2).sum(dim=1).unsqueeze(0) - 2 * self.Pos @ pos_mean.T  # (N, K)
        Pos_intra_homo = (P * torch.exp(-dist2 / (2.0 * self.sigma ** 2))).sum(dim=0) / (mass + self.eps)

        """ Weighted fusion of intra-parcel homogeneity """
        loss_homo = ((1-Pos_intra_homo) * self.tau).mean() + (1 - Feature_intra_homo).mean()
        # loss_homo = (1 - Feature_intra_homo).mean()

        """ Balance loss """
        probs = mass / (N + self.eps)
        loss_balance = (probs * torch.log(probs + self.eps)).sum()

        total_loss = loss_homo + loss_hete + 1 * loss_balance

        return total_loss, Feature_intra_homo.mean().item()

    @torch.no_grad()
    def update_tau(self, P):

        """

            Update the spatial weight: If decreasing the spatial weight does not cause the parcel to fragment, continue to decrease it. Conversely, if decreasing
            the weight leads to fragmentation, increase the spatial weight to ensure the parcel's contiguity.

        """

        N, K = P.shape

        pred_labels = torch.argmax(P, dim=1).cpu().numpy()
        edge_index_np = self.edge_index.cpu().numpy()

        row = edge_index_np[0]
        col = edge_index_np[1]
        data = np.ones_like(row)
        adj = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()

        for k in range(K):
            nodes_k = np.where(pred_labels == k)[0]
            if len(nodes_k) == 0:
                continue

            subgraph = adj[nodes_k, :][:, nodes_k]

            n_components, _ = connected_components(subgraph, directed=False)

            if n_components > 1:
                self.tau[k] = min(self.tau[k].item() * self.increase_factor, self.max_tau)
            else:
                self.tau[k] = max(self.tau[k].item() * self.decrease_factor, self.min_tau)



''' Loss Class 3: Spatial Smoothing Loss '''

''' Loss 4 Spatial continuity loss '''

class Spatial_Continuity_Loss(nn.Module):

    def __init__(self,individual_parcellation_edge_index):

        super().__init__()
        self.individual_parcellation_edge_index=individual_parcellation_edge_index

    def forward(self,assignment_output):

        probs = F.softmax(assignment_output, dim=1)
        src, dst = self.individual_parcellation_edge_index
        probs_src = probs[src]
        probs_dst = probs[dst]
        cos_sim = F.cosine_similarity(probs_src, probs_dst, dim=1)

        return (1.0 - cos_sim).mean()

''' Loss 5 Longitudinal consistency loss '''

class Longitudinal_Consistency_Loss(nn.Module):

    def __init__(self,age_list):
        super().__init__()

        self.weight = {}

        for target_age in age_list:

            for source_age in age_list:

                if source_age != target_age:

                    age_weight = np.exp(-(np.abs(target_age - source_age))/800)

                    self.weight[(str(target_age), str(source_age))] = age_weight


    def forward(self, assignment_outputs_dict):

        loss = 0.0
        ages = list(assignment_outputs_dict.keys())
        num_ages = len(ages)

        if num_ages <= 1:
            return torch.tensor(0.0, device=list(assignment_outputs_dict.values())[0].device)

        probs_dict = {age: F.softmax(logits, dim=1) for age, logits in assignment_outputs_dict.items()}

        total_pairs = 0

        for target_age in ages:
            target_logits = assignment_outputs_dict[target_age]
            target_probs = F.softmax(target_logits, dim=1)

            for source_age in ages:
                if source_age != target_age:

                    source_probs = probs_dict[source_age].detach()

                    cos_sim = F.cosine_similarity(target_probs, source_probs, dim=1)
                    pair_ce_loss = self.weight[(str(target_age), str(source_age))] * (1 - cos_sim)
                    pair_ce_loss = pair_ce_loss.mean()

                    loss += pair_ce_loss
                    total_pairs += 1

        return loss / total_pairs



