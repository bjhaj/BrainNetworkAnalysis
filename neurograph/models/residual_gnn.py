"""
ResidualGNNs Model for NeuroGraph
Original architecture from the NeuroGraph paper for HCP brain connectivity data
"""
import torch
from torch import nn
from torch.nn import ModuleList
from torch_geometric.nn import aggr
import torch.nn.functional as F

softmax = torch.nn.LogSoftmax(dim=1)


class ResidualGNNs(torch.nn.Module):
    """
    Residual Graph Neural Network architecture for brain connectomics.

    Combines flattened correlation matrix features with learned GNN representations
    through concatenation of all intermediate layer outputs.

    Args:
        args: Arguments containing model configuration (model type, num_classes)
        train_dataset: Training dataset to determine input dimensions
        hidden_channels: Hidden dimension for GNN layers
        hidden: Hidden dimension for MLP classifier
        num_layers: Number of GNN layers
        GNN: GNN layer class (GCNConv, GATConv, etc.)
        k: Unused parameter (kept for compatibility)
    """

    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features

        # Build GNN layers
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        # Calculate input dimensions for MLP
        # Upper triangular matrix flattened + GNN layer outputs
        input_dim1 = int(((num_features * num_features) / 2) - (num_features / 2) + (hidden_channels * num_layers))
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))

        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels * num_layers)

        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        """
        Forward pass combining flattened correlation matrix and GNN features.

        Args:
            data: PyG Data object with x, edge_index, batch

        Returns:
            Log-softmax predictions
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]

        # Apply GNN layers with tanh activation
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]

        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                # Flatten upper triangular of correlation matrix
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                # Mean pool GNN outputs
                xx = self.aggr(xx, batch)
                h.append(xx)

        # Concatenate all features
        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return softmax(x)
