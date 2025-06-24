import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def cosine_distance_torch(x, eps=1e-8):
    x_norm = x / (x.norm(dim=1, keepdim=True).clamp(min=eps))
    return 1 - torch.mm(x_norm, x_norm.t())

def get_all_mlp_weights(model):
    weights = []
    for i in tqdm(range(model.config.n_layer)):
        weights.append(model.transformer.h[i].mlp.c_proj.weight.detach().cpu())
    return torch.vstack(weights)

def compute_cluster_density(vectors, labels):
    cluster_densities = []
    for cluster_id in np.unique(labels):
        members = vectors[labels == cluster_id]
        if len(members) <= 1:
            continue
        pairwise = pdist(members.numpy(), metric='cosine')
        avg_dist = np.mean(pairwise)
        cluster_densities.append(avg_dist)
    return np.mean(cluster_densities)

def find_best_cluster_count(vectors, min_k=50, max_k=800, step=50):
    cosine_matrix = cosine_distance_torch(vectors).numpy()
    best_k = None
    best_density = float('inf')
    history = []

    for k in range(min_k, max_k + 1, step):
        clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
        labels = clustering.fit_predict(cosine_matrix)
        avg_density = compute_cluster_density(vectors, labels)
        history.append((k, avg_density))
        print(f"Clusters: {k}, Avg. Intra-cluster Cosine Distance: {avg_density:.4f}")
        if avg_density < best_density:
            best_density = avg_density
            best_k = k

    try:
        ks, densities = zip(*history)
        plt.plot(ks, densities, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Avg. Intra-cluster Cosine Distance')
        plt.title('Cluster Density Optimization')
        plt.grid(True)
        plt.savefig("cluster_density_plot.png")
        print("Saved plot to cluster_density_plot.png")
    except Exception:
        pass

    return best_k

if __name__ == "__main__":
    model_name = "distilgpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    vectors = get_all_mlp_weights(model)
    best_k = find_best_cluster_count(vectors, min_k=100, max_k=800, step=50)

    with open("optimal_num_clusters.pkl", "wb") as f:
        pickle.dump(best_k, f)
    print(f"Optimal number of clusters found: {best_k}")