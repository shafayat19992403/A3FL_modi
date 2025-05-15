from typing import List, Tuple, Dict, Union, List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
# from cuml.decomposition import PCA
# from cuml.cluster import DBSCAN



Scalar = Union[bool, bytes, float, int, str, List[int]]
Metrics = Dict[str, Scalar]
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define the function to extract client model weights and flatten them
# def extract_client_weights(client_models):
#     client_weights = []
#     for client_model in client_models:  # list of `Parameters` objects
#         weights = parameters_to_ndarrays(client_model)  # Convert Parameters to ndarray
#         flat_weights = np.concatenate([w.flatten() for w in weights])  # Flatten the weights
#         client_weights.append(flat_weights)
#     return client_weights

# def extract_client_weights(state_dicts: List[Dict[str,torch.Tensor]]):
#      """Turn a list of state_dicts into a list of 1D numpy vectors."""
#      flat_updates = []
#      for sd in state_dicts:
#          parts = []
#          for name, tensor in sd.items():
#              # skip tied embeddings or any buffers you don’t want to include
#              if 'decoder.weight' in name or '__' in name:
#                  continue
#              parts.append((tensor).flatten().cpu().numpy())
#          flat_updates.append(np.concatenate(parts, axis=0))
#      return flat_updates

def extract_client_weights(state_dicts: List[Dict[str, torch.Tensor]]):
    """Turn a list of state_dicts into normalized 1D numpy vectors."""
    flat_updates = []
    for sd in state_dicts:
        parts = []
        for name, tensor in sd.items():
            # skip tied embeddings or any buffers you don’t want to include
            if 'decoder.weight' in name or '__' in name:
                continue
            parts.append((tensor).flatten().cpu().numpy())
        flat_update = np.concatenate(parts, axis=0)

        # Normalize the update to unit length to make PCA/DBSCAN robust to non-IID scale differences
        # norm = np.linalg.norm(flat_update) + 1e-8  # avoid division by zero
        # flat_update = flat_update / norm

        flat_updates.append(flat_update)
    return flat_updates




# def apply_pca_to_weights(client_weights, client_ids, rnd, flagged_malicious_clients):
#     # client_weights is already numpy arrays here
#     pca = PCA(n_components=2)
#     print(client_weights)
#     reduced_weights = pca.fit_transform(client_weights)

#     # Extract PC1 values and reshape for clustering
#     pc1_values = reduced_weights[:, 0].reshape(-1, 1)
    
#     # Dynamic epsilon based on previous detections
#     eps_value = 1.2 if len(flagged_malicious_clients) > 0 else 1
    
#     # DBSCAN clustering
#     dbscan = DBSCAN(eps=eps_value, min_samples=2)
#     cluster_labels = dbscan.fit_predict(pc1_values)
#     # distance_matrix = cosine_distances(pc1_values)
#     # cluster_labels = dbscan.fit_predict(distance_matrix)
    
#     # Find outliers (malicious clients)
#     label_counts = Counter(cluster_labels)
#     outliers = []
#     print(label_counts, eps_value)
    
#     if len(label_counts) > 1:
#         smallest_cluster_size = min(label_counts.values())
#         outlier_labels = [label for label, count in label_counts.items() 
#                          if count == smallest_cluster_size]
#         outliers = [client_ids[i] for i, label in enumerate(cluster_labels) 
#                   if label in outlier_labels]
#     else:
#         outlier_labels = []
#         outliers = []

#     # Visualization (optional)
#     plt.figure(figsize=(10, 6))
#     plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], c=cluster_labels, cmap='viridis')
    
#     if outliers:
#         outlier_indices = [client_ids.index(client_id) for client_id in outliers]
#         plt.scatter(reduced_weights[outlier_indices, 0], 
#                    reduced_weights[outlier_indices, 1], 
#                    color='red', marker='x', s=100)
    
#     plt.title(f"Round {rnd}: PCA of Client Weights (Outliers: {outliers})")
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.colorbar(label='Cluster')
#     plt.savefig(f"pca_round_{rnd}.png")
#     plt.close()

#     return outliers, []


# def apply_pca_to_weights(client_weights, client_ids, rnd, flagged_malicious_clients):
#     """
#     Detects outliers among client weight updates using PCA + Mahalanobis distance.
    
#     Args:
#       client_weights: numpy.ndarray of shape (n_clients, n_features)
#       client_ids: list of client identifiers
#       rnd: current round number
#       flagged_malicious_clients: previously detected clients (unused here)
    
#     Returns:
#       outliers: list of client_ids flagged as outliers
#       Z: PCA-transformed coordinates of shape (n_clients, n_components)
#     """
#     # 1) Standardize features so each weight dimension has zero mean & unit variance
#     scaler = StandardScaler()
#     X = scaler.fit_transform(client_weights)

#     # 2) PCA with whitening
#     n_components = min(X.shape[0], X.shape[1])
#     pca = PCA(n_components=n_components, whiten=True)
#     Z = pca.fit_transform(X)  # shape: (n_clients, n_components)

#     # 3) Compute Mahalanobis distance in PCA space
#     cov = EmpiricalCovariance().fit(Z)
#     md2 = cov.mahalanobis(Z)    # squared Mahalanobis distances
#     dists = np.sqrt(md2)        # Euclidean distances in whitened PC space

#     # 4) Threshold = mean + 2*std, fallback to max if none exceed
#     mean_dist, std_dist = dists.mean(), dists.std()
#     thresh = mean_dist + 2 * std_dist
#     out_idx = np.where(dists > thresh)[0]
#     if len(out_idx) == 0:
#         out_idx = [int(np.argmax(dists))]

#     outliers = [client_ids[i] for i in out_idx]

#     # 5) Plot PC1 vs PC2 with annotations
#     plt.figure(figsize=(6,6))
#     plt.scatter(Z[:, 0], Z[:, 1], c='gray', s=80)
#     for i, (x, y) in enumerate(Z[:, :2]):
#         plt.text(x, y, str(client_ids[i]), fontsize=8, ha='center', va='center')

#     if outliers:
#         idxs = [client_ids.index(cid) for cid in outliers]
#         plt.scatter(Z[idxs, 0], Z[idxs, 1], color='red', marker='x', s=150, label='Outlier')
#         plt.legend()

#     plt.title(f"Round {rnd}: PCA of Client Weights (Outliers: {outliers})")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.savefig(f"pca_round_{rnd}.png")
#     plt.close()

#     return outliers, Z

def apply_pca_to_weights(client_weights, client_ids, rnd, flagged_malicious_clients):
    """
    Detects outliers among client weight updates using PCA + DBSCAN on PC1 only.
    
    Args:
      client_weights: numpy.ndarray of shape (n_clients, n_features)
      client_ids: list of client identifiers
      rnd: current round number
      flagged_malicious_clients: previously detected clients (unused here)
    
    Returns:
      outliers: list of client_ids flagged as outliers
      Z: PCA-transformed coordinates of shape (n_clients, n_components)
    """
    # 1) Standardize features so each weight dimension has zero mean & unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(client_weights)

    # 2) Full PCA (we'll still plot PC1 vs PC2 for context)
    n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, whiten=True)
    Z = pca.fit_transform(X)  # shape: (n_clients, n_components)

    # 3) Extract PC1 values for clustering
    pc1 = Z[:, 0].reshape(-1, 1)

    # 4) DBSCAN on just PC1
    #    eps can be tuned; here we choose 0.5 as a starting point
    eps_value = 0.5  
    db = DBSCAN(eps=eps_value, min_samples=2).fit(pc1)
    labels = db.labels_       # -1 = noise, 0,1,2... = clusters

    # 5) Identify the smallest cluster(s) (including noise as its own label)
    counts = Counter(labels)
    # pick the cluster label(s) with the fewest members
    smallest_size = min(counts.values())
    outlier_labels = [lbl for lbl, cnt in counts.items() if cnt == smallest_size]
    # map back to client IDs
    outliers = [client_ids[i] for i, lbl in enumerate(labels) if lbl in outlier_labels]

    # 6) Plot PC1 vs PC2, color by DBSCAN label and mark outliers
    plt.figure(figsize=(6,6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=80)
    plt.colorbar(scatter, label='DBSCAN cluster')

    for i, (x, y) in enumerate(Z[:, :2]):
        plt.text(x, y, str(client_ids[i]), fontsize=8, ha='center', va='center')

    if outliers:
        idxs = [client_ids.index(cid) for cid in outliers]
        plt.scatter(Z[idxs, 0], Z[idxs, 1],
                    color='red', marker='x', s=150, label='Outlier')
        plt.legend()

    plt.title(f"Round {rnd}: PCA (DBSCAN on PC1) → Outliers: {outliers}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(f"pca_round_{rnd}.png")
    plt.close()

    return outliers, Z
