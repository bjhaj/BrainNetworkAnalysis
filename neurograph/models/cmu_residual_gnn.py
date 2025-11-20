"""
Model wrapper to adapt ResidualGNNs for CMU Brain data
Keeps the same architecture, just handles input formatting differently
"""
import torch
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout, Sequential, ModuleList
from torch_geometric.nn import aggr, global_mean_pool
import torch.nn.functional as F

softmax = torch.nn.LogSoftmax(dim=1)


class CMUResidualGNNs(torch.nn.Module):
    """
    Adapted ResidualGNNs for CMU Brain data.
    Same architecture as NeuroGraph's ResidualGNNs, but handles node features directly
    instead of expecting adjacency matrix reconstruction.
    """

    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Get actual number of node features from dataset
        num_features = train_dataset[0].x.shape[1]

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

        # Calculate input dimension for MLP
        # We use mean-pooled node features + GNN layer outputs
        input_dim = num_features + (hidden_channels * num_layers)

        # MLP for classification
        self.mlp = Sequential(
            Linear(input_dim, hidden),
            BatchNorm1d(hidden),
            ReLU(),
            Dropout(0.5),
            Linear(hidden, hidden // 2),
            BatchNorm1d(hidden // 2),
            ReLU(),
            Dropout(0.5),
            Linear(hidden // 2, hidden // 2),
            BatchNorm1d(hidden // 2),
            ReLU(),
            Dropout(0.5),
            Linear(hidden // 2, args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Store initial features (mean-pooled across nodes)
        x_init = self.aggr(x, batch)

        # Apply GNN layers
        gnn_outputs = []
        xs = [x]
        for conv in self.convs:
            xs.append(conv(xs[-1], edge_index).tanh())

        # Pool GNN outputs
        for i, xx in enumerate(xs):
            if i > 0:  # Skip initial features (already have x_init)
                pooled = self.aggr(xx, batch)
                gnn_outputs.append(pooled)

        # Concatenate: [initial_features, gnn_layer_1, gnn_layer_2, ...]
        if len(gnn_outputs) > 0:
            h = torch.cat([x_init] + gnn_outputs, dim=1)
        else:
            h = x_init

        # Apply MLP
        out = self.mlp(h)

        return softmax(out)


class CMUResidualGNNsSimple(torch.nn.Module):
    """
    Simplified version - just GNN + global pooling + MLP
    Even more minimal adaptation
    """

    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN):
        super().__init__()
        self.convs = ModuleList()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        num_features = train_dataset[0].x.shape[1]

        # Build GNN layers
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        # Final layer dimension (after all GNN layers)
        final_dim = hidden_channels if num_layers > 0 else num_features

        # Simple MLP
        self.mlp = Sequential(
            Linear(final_dim, hidden),
            BatchNorm1d(hidden),
            ReLU(),
            Dropout(0.5),
            Linear(hidden, args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Classification
        out = self.mlp(x)

        return F.log_softmax(out, dim=1)


# Export the models
__all__ = ['CMUResidualGNNs', 'CMUResidualGNNsSimple']
