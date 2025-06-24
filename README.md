# LLM_Unlearn

A toolkit for analyzing, clustering, and potentially unlearning knowledge from internal components of Large Language Models (LLMs) like GPT-2. This repo supports neuron-level introspection, semantic clustering, and interpretability via token projection patterns.

---

## 📺 Installation

```bash
git clone https://github.com/Sahil-R-Kale/LLM_Unlearn.git
cd LLM_Unlearn
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Find Optimal Number of Clusters
```bash
python get_optimal_clusters.py
```

### 2. Generate Cluster Mapping Files
```bash
python project_vectors.py
```

These will produce:
- `optimal_num_clusters.pkl` — Best cluster count based on density.
- `cluster_to_value.pkl` — Cluster ID → Neuron list.
- `value_to_cluster.pkl` — Neuron (layer, index) → Cluster ID.

---

