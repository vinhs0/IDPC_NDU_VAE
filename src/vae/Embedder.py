import numpy as np
import torch
from torch_geometric.nn import Node2Vec

class GraphEmbedder:
    def __init__(self, embedding_dim=16, epochs=10):
        self.embedding_dim = embedding_dim
        self.epochs = epochs

    def _convert_NDList2Graph(self, chromosome):
        """
        PRIVATE METHOD: The outside world doesn't need to call this.
        It handles the annoying translation to COO format internally.
        """
        x = []
        edge_sources = []
        edge_targets = []
        
        for i, nd in enumerate(chromosome):
            x.append([float(nd.node), float(nd.depth)])
            if i > 0:
                for j in range(i - 1, -1, -1):
                    if chromosome[j].depth == nd.depth - 1:
                        edge_sources.extend([j, i])
                        edge_targets.extend([i, j])
                        break
                        
        return np.array(x, dtype=np.float32), np.array([edge_sources, edge_targets], dtype=np.int64)

    def get_embeddings(self, chromosome):
        """
        PUBLIC METHOD: This is the only thing your VAE or Main Loop calls.
        Input: List[NodeDepth]
        Output: PyTorch Tensors (node_features, edge_index)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Lấy COO List
        x_np, edge_index_np = self._convert_NDList2Graph(chromosome)
        
        # 2. Convert to Tensors
        x = torch.tensor(x_np, dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        
        if edge_index.numel() == 0:
            zero_padding = torch.zeros((x.shape[0], self.embedding_dim))
            return torch.cat([x, zero_padding], dim=1), edge_index

        # 3. Train Node2Vec
        model = Node2Vec(
            edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=20, context_size=10, walks_per_node=10,
            num_negative_samples=1, p=1.0, q=1.0, sparse=True
        ).to(device)

        loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        model.train()
        for epoch in range(self.epochs):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

        # 4. Extract and combine
        model.eval()
        with torch.no_grad():
            structural_embeddings = model().cpu()

        combined_features = torch.cat([x, structural_embeddings], dim=1)

        # Return the final clean tensors
        return combined_features, edge_index