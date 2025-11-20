"""
CMU Brain Dataset Adapter for NeuroGraph GNN Architecture
Minimal wrapper to use existing ResidualGNNs with CMU data
"""
import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from pathlib import Path

# Import CMU data loaders from the data module
from neurograph.data import BrainDataLoader


class CMUBrainDataset(InMemoryDataset):
    """
    CMU Brain dataset wrapper compatible with NeuroGraph interface.
    Mimics the structure of NeuroGraphDataset for seamless integration.
    """

    def __init__(self, root='data/CMUBrain', task='sex',
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root: Root directory containing raw/ folder
            task: Which task to load labels for ('sex', 'math', 'creativity')
            transform: Optional transform
            pre_transform: Optional pre-transform
            pre_filter: Optional pre-filter
        """
        self.task = task
        self.raw_data_dir = Path(root) / 'raw' / 'brainnetworks'

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        """Files that should exist in raw/brainnetworks/"""
        return ['metainfo.csv', 'smallgraphs']

    @property
    def processed_file_names(self):
        """Processed file name"""
        return [f'cmu_brain_{self.task}.pt']

    def download(self):
        """Data should already be in raw/ folder"""
        if not (self.raw_data_dir / 'metainfo.csv').exists():
            raise FileNotFoundError(
                f"metainfo.csv not found in {self.raw_data_dir}. "
                "Please ensure CMU Brain data is in data/CMUBrain/raw/brainnetworks/"
            )

    def process(self):
        """Process raw CMU data into PyG format"""
        print(f"\n{'='*60}")
        print(f"Processing CMU Brain Dataset - Task: {self.task}")
        print(f"{'='*60}\n")

        # Load data using CMU data loader
        loader = BrainDataLoader(str(self.raw_data_dir))
        adj_list, feat_list, labels = loader.load_all_subjects(
            symmetrize=True,
            add_self_loops=False,  # Will be added by GNN if needed
            threshold=0.0,
            log_scale=True  # Log-transform edge weights
        )

        # Create PyG Data objects
        data_list = []

        # Determine label based on task
        if self.task == 'sex':
            # Binary classification: 0=Female, 1=Male
            task_labels = labels['sex']
            num_classes = 2
        elif self.task == 'math':
            # Regression on FSIQ scores
            task_labels = labels['math']
            # For consistency with classification interface, we'll bin into classes
            # or keep as regression - let's do binary classification on median split
            median_fsiq = np.nanmedian(task_labels[~np.isnan(task_labels)])
            task_labels = (task_labels >= median_fsiq).astype(int)
            num_classes = 2
        elif self.task == 'creativity':
            # Regression on CAQ scores - binary on median split
            task_labels = labels['creativity']
            median_caq = np.nanmedian(task_labels[~np.isnan(task_labels)])
            task_labels = (task_labels >= median_caq).astype(int)
            num_classes = 2
        else:
            raise ValueError(f"Unknown task: {self.task}")

        print(f"Creating PyG Data objects for {len(adj_list)} subjects...")

        for idx, (A, X) in enumerate(zip(adj_list, feat_list)):
            # Skip if label is NaN
            if np.isnan(task_labels[idx]):
                continue

            # Convert adjacency to edge_index
            edge_index, edge_weight = self._adj_to_edge_index(A)

            # Create Data object
            data = Data(
                x=torch.FloatTensor(X),
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=torch.LongTensor([task_labels[idx]]),
                num_nodes=X.shape[0]
            )

            # Apply filters
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        print(f"Created {len(data_list)} valid graphs")
        print(f"Node features: {data_list[0].x.shape}")
        print(f"Number of edges: {data_list[0].edge_index.shape[1]}")
        print(f"Label distribution: {np.bincount([d.y.item() for d in data_list])}")

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        print(f"\nSaved processed data to: {self.processed_paths[0]}")
        print(f"{'='*60}\n")

    def _adj_to_edge_index(self, A: np.ndarray):
        """Convert adjacency matrix to edge_index and edge_weight"""
        # Find non-zero edges
        row, col = np.where(A > 0)
        edge_weight = A[row, col]

        # Create edge_index (COO format)
        edge_index = torch.LongTensor(np.vstack([row, col]))
        edge_weight = torch.FloatTensor(edge_weight)

        return edge_index, edge_weight


if __name__ == '__main__':
    """Test the dataset adapter"""
    print("\n" + "="*60)
    print("Testing CMU Brain Dataset Adapter")
    print("="*60 + "\n")

    # Test loading each task
    for task in ['sex', 'math', 'creativity']:
        print(f"\nLoading task: {task}")
        print("-" * 40)

        dataset = CMUBrainDataset(root='data/CMUBrain', task=task)

        print(f"Dataset size: {len(dataset)}")
        print(f"Number of features: {dataset.num_features}")
        print(f"Number of classes: {dataset.num_classes}")

        # Test a sample
        sample = dataset[0]
        print(f"Sample graph:")
        print(f"  Nodes: {sample.num_nodes}")
        print(f"  Edges: {sample.edge_index.shape[1]}")
        print(f"  Node features: {sample.x.shape}")
        print(f"  Label: {sample.y.item()}")

        # Check label distribution
        labels = [d.y.item() for d in dataset]
        print(f"Label distribution: {np.bincount(labels)}")
        print()
