import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import Graph

class ConvTemporalGraphical(nn.Module):
    """
    Spatial Graph Convolution layer.
    Computes graph convolution over spatial partitions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x, A):
        # x: (N * M, C_in, T, V)
        # A: (kernel_size, V, V)
        assert A.size(0) == self.kernel_size

        # Temporal/spatial convolution on 1x1 kernel
        x = self.conv(x)  # (N * M, C_out * kernel_size, T, V)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v) # (N*M, kernel_size, C_out, T, V)
        
        # Graph convolution via einsum (summing over partition subsets and contracting vertices)
        # x: (n, k, c, t, v)
        # A: (k, v, w) -> where v is source vertex, w is target vertex
        # out: (n, c, t, w)
        out = torch.einsum('nkctv,kvw->nctw', x, A)
        return out.contiguous()


class ST_GCN_Block(nn.Module):
    """
    Spatio-Temporal Graph Convolution Block.
    Contains: Spatial Graph Conv -> TCN (Temporal Convolution) -> Residual -> ReLU & Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_node, stride=1, dropout=0.5, residual=True):
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)
        
        # Learnable edge importance weight parameter
        self.edge_importance = nn.Parameter(torch.ones(kernel_size, num_node, num_node))
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),  # Padding to preserve sequence length when stride=1
            ),
            nn.BatchNorm2d(out_channels)
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, x, A):
        # Apply GCN with learnable edge weights
        gcn_out = self.gcn(x, A * self.edge_importance)
        
        # Apply TCN and add residual connection
        x = self.tcn(gcn_out) + self.residual(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x, A


class STGCN(nn.Module):
    """
    Spatial Temporal Graph Convolutional Network (ST-GCN)
    for skeleton-based action / fall detection.
    """
    def __init__(self, in_channels=3, num_classes=2, graph_args=None, edge_importance_weighting=True):
        super().__init__()
        
        if graph_args is None:
            graph_args = {'layout': 'coco', 'strategy': 'spatial'}
            
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        # Graph parameters
        spatial_kernel_size = A.size(0)
        num_node = A.size(1)
        
        # 1. Input Normalization: BatchNorm over (Joints * Coordinates)
        self.data_bn = nn.BatchNorm1d(in_channels * num_node)
        
        # 2. ST-GCN Stack
        # Block 1, 2, 3: 64 output channels
        self.st_gcn1 = ST_GCN_Block(in_channels, 64, spatial_kernel_size, num_node, stride=1, residual=False)
        self.st_gcn2 = ST_GCN_Block(64, 64, spatial_kernel_size, num_node, stride=1)
        self.st_gcn3 = ST_GCN_Block(64, 64, spatial_kernel_size, num_node, stride=1)
        
        # Block 4, 5, 6: 128 output channels, Block 4 downsamples temporally
        self.st_gcn4 = ST_GCN_Block(64, 128, spatial_kernel_size, num_node, stride=2)
        self.st_gcn5 = ST_GCN_Block(128, 128, spatial_kernel_size, num_node, stride=1)
        self.st_gcn6 = ST_GCN_Block(128, 128, spatial_kernel_size, num_node, stride=1)
        
        # Block 7, 8, 9: 256 output channels, Block 7 downsamples temporally
        self.st_gcn7 = ST_GCN_Block(128, 256, spatial_kernel_size, num_node, stride=2)
        self.st_gcn8 = ST_GCN_Block(256, 256, spatial_kernel_size, num_node, stride=1)
        self.st_gcn9 = ST_GCN_Block(256, 256, spatial_kernel_size, num_node, stride=1)
        
        # 3. Classifier Head
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input x shape: (N, C, T, V, M)
        # N: Batch size, C: Channels, T: Seq length, V: Num nodes, M: Num persons
        N, C, T, V, M = x.size()
        
        # Input Normalization
        # Reshape to (N * M, V * C, T) for BatchNorm1d
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        
        # Reshape back to (N * M, C, T, V)
        x = x.view(N * M, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()      # (N * M, C, T, V)
        
        # Forward through ST-GCN blocks
        x, _ = self.st_gcn1(x, self.A)
        x, _ = self.st_gcn2(x, self.A)
        x, _ = self.st_gcn3(x, self.A)
        
        x, _ = self.st_gcn4(x, self.A)
        x, _ = self.st_gcn5(x, self.A)
        x, _ = self.st_gcn6(x, self.A)
        
        x, _ = self.st_gcn7(x, self.A)
        x, _ = self.st_gcn8(x, self.A)
        x, _ = self.st_gcn9(x, self.A)
        
        # Global pooling over time (T) and space (V)
        # Shape of x: (N * M, 256, T_out, V)
        x = F.avg_pool2d(x, x.size()[2:])  # (N * M, 256, 1, 1)
        
        # Reshape to (N, M, 256)
        x = x.view(N, M, -1)
        
        # Mean pooling over M (persons) -> (N, 256)
        x = x.mean(dim=1)
        
        # Classification layer -> (N, num_classes)
        logits = self.fc(x)
        return logits
