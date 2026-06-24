import numpy as np

class Graph:
    """
    Skeletal Graph representation for ST-GCN.
    Defines joints, links, and builds normalized adjacency matrices.
    """
    def __init__(self, layout='coco', strategy='spatial'):
        self.layout = layout
        self.strategy = strategy
        self.get_edge()
        self.get_adjacency()

    def get_edge(self):
        if self.layout == 'coco':
            self.num_node = 17
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.neighbor_link = [
                (0, 1), (0, 2), (1, 3), (2, 4),             # Head area
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),     # Arms / Upper Torso
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Legs / Lower Torso
                (5, 11), (6, 12)                            # Torso sides
            ]
            self.edge = self.self_link + self.neighbor_link
            
            # Static distance rank relative to the body's center of gravity (torso)
            # Lower rank is closer to torso, higher rank is outer extremities.
            self.part_rank = [
                1,  # 0: nose
                2,  # 1: left eye
                2,  # 2: right eye
                3,  # 3: left ear
                3,  # 4: right ear
                0,  # 5: left shoulder
                0,  # 6: right shoulder
                1,  # 7: left elbow
                1,  # 8: right elbow
                2,  # 9: left wrist
                2,  # 10: right wrist
                0,  # 11: left hip
                0,  # 12: right hip
                1,  # 13: left knee
                1,  # 14: right knee
                2,  # 15: left ankle
                2   # 16: right ankle
            ]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_adjacency(self):
        if self.strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            for i, j in self.edge:
                A[0, i, j] = 1
                A[0, j, i] = 1
            self.A = self.normalize_adjacency(A)
            
        elif self.strategy == 'spatial':
            # Partition into 3 subsets:
            # Subset 0: Self-loops (joints connected to themselves)
            # Subset 1: Centripetal movement (directed towards the center of gravity)
            # Subset 2: Centrifugal movement (directed away from the center of gravity)
            A = np.zeros((3, self.num_node, self.num_node))
            for i in range(self.num_node):
                A[0, i, i] = 1
                
            for i, j in self.neighbor_link:
                if self.part_rank[i] < self.part_rank[j]:
                    A[1, j, i] = 1  # Centripetal: j -> i (higher rank to lower rank)
                    A[2, i, j] = 1  # Centrifugal: i -> j (lower rank to higher rank)
                elif self.part_rank[i] > self.part_rank[j]:
                    A[1, i, j] = 1  # Centripetal: i -> j
                    A[2, j, i] = 1  # Centrifugal: j -> i
                else:
                    # Same rank (e.g. shoulder to shoulder or hip to hip), assign symmetrically to centripetal
                    A[1, i, j] = 1
                    A[1, j, i] = 1
            self.A = self.normalize_adjacency(A)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def normalize_adjacency(self, A):
        # A: (num_subsets, num_node, num_node)
        # Normalize using D^(-1/2) * A * D^(-1/2) for each subset
        normalized_A = np.zeros_like(A)
        for i in range(A.shape[0]):
            adj = A[i]
            rowsum = np.sum(adj, axis=1)
            d_inv_sqrt = np.zeros_like(rowsum, dtype=float)
            mask = rowsum > 0
            d_inv_sqrt[mask] = rowsum[mask] ** -0.5
            d_mat_inv_sqrt = np.diag(d_inv_sqrt)
            normalized_A[i] = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return normalized_A
