"""Drug-only attention readout and layer gating; no cell-conditioned graph attention."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax

VARIANTS = ['baseline', 'attn_pool', 'layer_gate', 'full', 'equal_fusion',
            'capacity', 'gcn', 'gatv2', 'drug_only', 'cell_only']


class AttentionReadout(nn.Module):
    def __init__(self, width=512, hidden=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, batch, return_weights=False):
        weight = softmax(self.score(x).view(-1), batch)
        out = global_add_pool(x * weight.unsqueeze(-1), batch)
        return (out, weight) if return_weights else out


class MedicalGAT(nn.Module):
    def __init__(self, cell_dim, variant='full', width=512, heads=2, hidden=64):
        super().__init__()
        if variant not in VARIANTS or width % heads:
            raise ValueError('Unsupported model configuration')
        self.variant = variant
        self.multilevel = variant in ('layer_gate', 'full', 'equal_fusion')
        self.attentive = variant in ('attn_pool', 'full', 'equal_fusion')
        self.capacity_extra_parameters = None
        self.capacity_target_parameters = None
        if variant != 'cell_only':
            if variant == 'gcn':
                self.gat1, self.gat2 = GCNConv(768, width), GCNConv(width, width)
            else:
                conv = GATv2Conv if variant == 'gatv2' else GATConv
                self.gat1 = conv(768, width // heads, heads=heads, dropout=0.1)
                self.gat2 = conv(width, width // heads, heads=heads, dropout=0.1)
            if self.attentive:
                self.pool2 = AttentionReadout(width, hidden)
                if self.multilevel:
                    self.pool1 = AttentionReadout(width, hidden)
            if variant in ('layer_gate', 'full'):
                self.layer_gate = nn.Sequential(nn.Linear(2 * width, hidden), nn.Tanh(),
                                                nn.Linear(hidden, 2))
            if variant == 'capacity':
                # Match full-model extra parameters before seeing any response metrics.
                pool_params = (width * hidden + hidden + hidden + 1)
                gate_params = 2 * width * hidden + hidden + 2 * hidden + 2
                target = 2 * pool_params + gate_params
                middle = max(1, round((target - width) / (2 * width + 1)))
                self.capacity_mlp = nn.Sequential(nn.Linear(width, middle), nn.ReLU(),
                                                  nn.Linear(middle, width))
                self.capacity_extra_parameters = sum(p.numel() for p in self.capacity_mlp.parameters())
                self.capacity_target_parameters = target
            self.drug_fc = nn.Sequential(nn.Linear(width, 1024), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(1024, 256), nn.ReLU())
        if variant != 'drug_only':
            self.cell_fc = nn.Sequential(nn.Linear(cell_dim, 1024), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(512, 256), nn.ReLU())
        pair_width = 256 if variant in ('drug_only', 'cell_only') else 512
        self.predict_fc = nn.Sequential(nn.Linear(pair_width, 1024), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(512, 1))

    def drug_vector(self, data, explain=False):
        h1 = F.elu(self.gat1(data.x, data.edge_index))
        h2 = F.elu(self.gat2(h1, data.edge_index))
        details = {}
        if self.attentive:
            p2, a2 = self.pool2(h2, data.batch, True)
            details['atom_weights_layer2'] = a2
        else:
            p2 = global_mean_pool(h2, data.batch)
        if self.multilevel:
            if self.attentive:
                p1, a1 = self.pool1(h1, data.batch, True)
                details['atom_weights_layer1'] = a1
            else:
                p1 = global_mean_pool(h1, data.batch)
            if self.variant == 'equal_fusion':
                gates = torch.full((p1.shape[0], 2), 0.5, device=p1.device)
            else:
                gates = self.layer_gate(torch.cat([p1, p2], dim=1)).softmax(dim=1)
            pooled = gates[:, :1] * p1 + gates[:, 1:] * p2
            details['layer_weights'] = gates
        else:
            pooled = p2
        if self.variant == 'capacity':
            pooled = pooled + self.capacity_mlp(pooled)
        out = self.drug_fc(pooled)
        return (out, details) if explain else out

    def forward(self, data):
        if self.variant == 'cell_only':
            combined = self.cell_fc(data.GEF)
        elif self.variant == 'drug_only':
            combined = self.drug_vector(data)
        else:
            combined = torch.cat([self.drug_vector(data), self.cell_fc(data.GEF)], dim=1)
        return self.predict_fc(combined).view(-1)
