import torch, numpy as np
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj

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
        x = self.proj(x)
        F_dim = x.size(1)
        out = torch.zeros_like(x)
        for g in range(batch.max().item() + 1):
            mask = (batch == g)
            x_g = x[mask]
            n = x_g.size(0)
            node_ids = mask.nonzero().squeeze()
            if node_ids.dim() == 0:
                node_ids = node_ids.unsqueeze(0)
            local_ei = edge_index[:, torch.isin(edge_index[0], node_ids) &
                                     torch.isin(edge_index[1], node_ids)]
            id_map = {nid.item(): i for i, nid in enumerate(node_ids)}
            if local_ei.numel() > 0:
                le = torch.tensor([[id_map[e.item()] for e in local_ei[0]],
                                   [id_map[e.item()] for e in local_ei[1]]])
                adj = to_dense_adj(le, max_num_nodes=n)[0]
            else:
                adj = torch.zeros(n, n)
            H = self.strength * (adj + adj.T) / 2
            H = H.unsqueeze(0)
            q = x_g.unsqueeze(0)
            p = torch.zeros_like(q)
            q_out_list = []
            for f in range(F_dim):
                qf, _ = self.leapfrog(q[:,:,f:f+1], p[:,:,f:f+1], H)
                q_out_list.append(qf)
            out[mask] = torch.cat(q_out_list, dim=-1).squeeze(0)
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
        correct = sum((model(d).argmax(1) == d.y).sum().item() for d in test_loader)
        total = sum(len(d.y) for d in test_loader)
        best = max(best, correct/total)
    return best

dataset = TUDataset(root='/tmp/proteins', name='PROTEINS')
print(f"PROTEINS: {len(dataset)} graphs, {dataset.num_classes} classes, "
      f"{dataset.num_node_features} node features")

seeds = [42, 7, 13, 99, 123]
gcn_accs, hgnn_accs = [], []

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = dataset.shuffle()
    split = int(0.8 * len(ds))
    tr = DataLoader(ds[:split], batch_size=32, shuffle=True)
    te = DataLoader(ds[split:], batch_size=32)
    gcn_acc  = train_eval(StandardGCN(dataset.num_node_features, 32, dataset.num_classes), tr, te)
    hgnn_acc = train_eval(HamiltonianGNN(dataset.num_node_features, 32, dataset.num_classes), tr, te)
    gcn_accs.append(gcn_acc)
    hgnn_accs.append(hgnn_acc)
    print(f"seed {seed:5d} | GCN {gcn_acc:.3f} | HamGNN {hgnn_acc:.3f}", flush=True)

print(f"\n{'='*45}")
print(f"GCN     mean {np.mean(gcn_accs):.3f} ± {np.std(gcn_accs):.3f}")
print(f"HamGNN  mean {np.mean(hgnn_accs):.3f} ± {np.std(hgnn_accs):.3f}")
print(f"{'='*45}")
