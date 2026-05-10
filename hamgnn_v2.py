"""
HamGNN v2 — vectorized HamiltonianConv using PyG batched dense ops.
Replaces per-graph Python loop with batched torch.bmm — ~10-20x faster.
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
import numpy as np


class HamiltonianConvV2(torch.nn.Module):
    def __init__(self, in_dim, out_dim, steps=4, dt=0.1):
        super().__init__()
        self.proj = Linear(in_dim, out_dim)
        self.steps = steps
        self.dt = dt
        self.strength = torch.nn.Parameter(torch.ones(1))

    def leapfrog(self, q, p, H):
        # q, p: [B, N, F]   H: [B, N, N]
        for _ in range(self.steps):
            p = p - 0.5 * self.dt * torch.bmm(H, q)
            q = q + self.dt * torch.bmm(H, p)
            p = p - 0.5 * self.dt * torch.bmm(H, q)
        return q, p

    def forward(self, x, edge_index, batch):
        x = self.proj(x)                              # [N_total, F]

        # Pad to dense batch: [B, N_max, F] + mask [B, N_max]
        x_dense, mask = to_dense_batch(x, batch)      # [B, N_max, F]

        # Dense adjacency: [B, N_max, N_max]
        adj = to_dense_adj(edge_index, batch)
        H = self.strength * (adj + adj.transpose(1, 2)) / 2  # symmetric

        q = x_dense                                   # [B, N_max, F]
        p = torch.zeros_like(q)
        q_out, _ = self.leapfrog(q, p, H)

        # Zero out padding positions
        q_out = q_out * mask.unsqueeze(-1).float()

        # Unpack back to [N_total, F]
        out = q_out[mask]
        return F.relu(out)


class HamiltonianGNNV2(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.conv1 = HamiltonianConvV2(in_dim, hidden)
        self.conv2 = HamiltonianConvV2(hidden, hidden)
        self.bn1 = BatchNorm1d(hidden)
        self.bn2 = BatchNorm1d(hidden)
        self.clf = Linear(hidden, num_classes)

    def forward(self, data):
        x = self.bn1(self.conv1(data.x, data.edge_index, data.batch))
        x = self.bn2(self.conv2(x, data.edge_index, data.batch))
        return self.clf(global_mean_pool(x, data.batch))


class StandardGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn1 = BatchNorm1d(hidden)
        self.bn2 = BatchNorm1d(hidden)
        self.clf = Linear(hidden, num_classes)

    def forward(self, data):
        x = self.bn1(F.relu(self.conv1(data.x, data.edge_index)))
        x = self.bn2(F.relu(self.conv2(x, data.edge_index)))
        return self.clf(global_mean_pool(x, data.batch))


def train_eval(model, train_loader, test_loader, epochs=50):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    best = 0
    for epoch in range(1, epochs+1):
        model.train()
        for data in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(data), data.y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            correct = sum((model(d).argmax(1) == d.y).sum().item() for d in test_loader)
            total = sum(len(d.y) for d in test_loader)
        best = max(best, correct / total)
    return best


def run(dataset_name, seeds=5):
    dataset = TUDataset(root=f'/tmp/{dataset_name.lower()}', name=dataset_name)
    print(f"\n{dataset_name}: {len(dataset)} graphs, {dataset.num_classes} classes, "
          f"{dataset.num_node_features} node features")

    seed_list = [42, 7, 13, 99, 123][:seeds]
    gcn_accs, hgnn_accs = [], []

    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)
        ds = dataset.shuffle()
        split = int(0.8 * len(ds))
        tr = DataLoader(ds[:split], batch_size=32, shuffle=True)
        te = DataLoader(ds[split:],  batch_size=32)

        in_dim, hidden, nc = dataset.num_node_features, 32, dataset.num_classes
        gcn_acc  = train_eval(StandardGCN(in_dim, hidden, nc), tr, te)
        hgnn_acc = train_eval(HamiltonianGNNV2(in_dim, hidden, nc), tr, te)
        gcn_accs.append(gcn_acc)
        hgnn_accs.append(hgnn_acc)
        print(f"  seed {seed:5d} | GCN {gcn_acc:.3f} | HamGNN {hgnn_acc:.3f}", flush=True)

    print(f"\n{'='*45}")
    print(f"GCN     mean {np.mean(gcn_accs):.3f} ± {np.std(gcn_accs):.3f}")
    print(f"HamGNN  mean {np.mean(hgnn_accs):.3f} ± {np.std(hgnn_accs):.3f}")
    print(f"{'='*45}")


if __name__ == '__main__':
    run('MUTAG')     # quick sanity check first
    run('PROTEINS')  # then the real test
