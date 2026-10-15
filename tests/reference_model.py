import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATNet(nn.Module):
    """Same architecture as the existing GAT baseline; only cell input size varies."""
    def __init__(self, drug_in_channels=768, cell_in_channels=5757,
                 hidden_dim=256, heads=2):
        super().__init__()
        self.gat1 = GATConv(drug_in_channels, hidden_dim, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1)
        self.drug_fc = nn.Sequential(nn.Linear(hidden_dim * heads, 1024), nn.ReLU(),
                                     nn.Dropout(0.1), nn.Linear(1024, 256), nn.ReLU())
        self.cell_fc = nn.Sequential(nn.Linear(cell_in_channels, 1024), nn.ReLU(),
                                     nn.Dropout(0.1), nn.Linear(1024, 512), nn.ReLU(),
                                     nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU())
        self.predict_fc = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(512, 1))

    def forward(self, data):
        x = F.elu(self.gat1(data.x, data.edge_index))
        x = F.elu(self.gat2(x, data.edge_index))
        drug = self.drug_fc(global_mean_pool(x, data.batch))
        cell = self.cell_fc(data.GEF)
        return self.predict_fc(torch.cat([drug, cell], dim=1)).view(-1)
