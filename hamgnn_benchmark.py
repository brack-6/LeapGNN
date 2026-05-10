"""
Hamiltonian GNN vs standard GNN benchmark on MUTAG dataset.
MUTAG: 188 graphs, 2 classes (mutagenic/non-mutagenic molecules).
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj
import numpy as np

# ── Hamiltonian GNN ──────────────────────────────────────────────────────────

class HamiltonianConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, steps=4, dt=0.1):
        super().__init__()
        self.proj = Linear(in_dim, out_dim)
        self.steps = steps
        self.dt = dt
        self.strength = torch.nn.Parameter(torch.ones(1))

    def leapfrog(self, q, p, H):
        for _ in range(self.steps):
            p = p - 0.5 * self.dt * torch.bmm(H, q)
            q = q + self.dt * torch.bmm(H, p)
            p = p - 0.5 * self.dt * torch.bmm(H, q)
        return q, p

    def forward(self, x, edge_index, batch):
        # Project features
        x = self.proj(x)                          # [N, F]
        num_graphs = batch.max().item() + 1
        F_dim = x.size(1)
        out = torch.zeros_like(x)

        for g in range(num_graphs):
            mask = (batch == g)
            x_g = x[mask]                         # [n, F]
            n = x_g.size(0)

            # Build adjacency for this graph
            src = edge_index[0][torch.isin(edge_index[0], mask.nonzero().squeeze())]
            # Use to_dense_adj scoped to graph
            node_ids = mask.nonzero().squeeze()
            local_edge_index = edge_index[:, torch.isin(edge_index[0], node_ids) &
                                             torch.isin(edge_index[1], node_ids)]
            # Remap to local indices
            id_map = {nid.item(): i for i, nid in enumerate(node_ids)}
            if local_edge_index.numel() > 0:
                le = torch.tensor([[id_map[e.item()] for e in local_edge_index[0]],
                                   [id_map[e.item()] for e in local_edge_index[1]]])
                adj = to_dense_adj(le, max_num_nodes=n)[0]  # [n, n]
            else:
                adj = torch.zeros(n, n)

            H = self.strength * (adj + adj.T) / 2   # symmetric Hamiltonian
            H = H.unsqueeze(0)                       # [1, n, n]
            q = x_g.unsqueeze(0)                     # [1, n, F]
            p = torch.zeros_like(q)

            # Leapfrog per feature dimension
            q_out_list = []
            for f in range(F_dim):
                qf = q[:, :, f:f+1]
                pf = p[:, :, f:f+1]
                qf, _ = self.leapfrog(qf, pf, H)
                q_out_list.append(qf)
            q_out = torch.cat(q_out_list, dim=-1)    # [1, n, F]
            out[mask] = q_out.squeeze(0)

        return F.relu(out)


class HamiltonianGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.conv1 = HamiltonianConv(in_dim, hidden)
        self.conv2 = HamiltonianConv(hidden, hidden)
        self.bn1 = BatchNorm1d(hidden)
        self.bn2 = BatchNorm1d(hidden)
        self.clf = Linear(hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn1(self.conv1(x, edge_index, batch))
        x = self.bn2(self.conv2(x, edge_index, batch))
        x = global_mean_pool(x, batch)
        return self.clf(x)


# ── Standard GCN ────────────────────────────────────────────────────────────

class StandardGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn1 = BatchNorm1d(hidden)
        self.bn2 = BatchNorm1d(hidden)
        self.clf = Linear(hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.clf(x)


# ── Training ─────────────────────────────────────────────────────────────────

def train_eval(model, train_loader, test_loader, epochs=50):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    best_acc = 0
    for epoch in range(1, epochs+1):
        model.train()
        for data in train_loader:
            opt.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            model.eval()
            correct = sum((model(d).argmax(1) == d.y).sum().item() for d in test_loader)
            total = sum(len(d.y) for d in test_loader)
            acc = correct / total
            best_acc = max(best_acc, acc)
            print(f"  epoch {epoch:3d} | acc {acc:.3f}")
    return best_acc


def run(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TUDataset(root='/tmp/mutag', name='MUTAG').shuffle()
    print(f"MUTAG: {len(dataset)} graphs, {dataset.num_classes} classes, "
          f"{dataset.num_node_features} node features")

    split = int(0.8 * len(dataset))
    train_ds, test_ds = dataset[:split], dataset[split:]
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32)

    in_dim = dataset.num_node_features
    hidden = 32

    print("\n── Standard GCN ──")
    gcn = StandardGCN(in_dim, hidden, dataset.num_classes)
    gcn_acc = train_eval(gcn, train_loader, test_loader)

    print("\n── Hamiltonian GNN ──")
    hgnn = HamiltonianGNN(in_dim, hidden, dataset.num_classes)
    hgnn_acc = train_eval(hgnn, train_loader, test_loader)

    print(f"\n{'='*35}")
    print(f"GCN best acc:          {gcn_acc:.3f}")
    print(f"Hamiltonian GNN acc:   {hgnn_acc:.3f}")
    print(f"{'='*35}")

if __name__ == '__main__':
    run()
