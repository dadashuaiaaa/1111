import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

# 设置量子设备
dev = qml.device('default.qubit', wires=9)

# 定义量子图卷积层
@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_graph_layer_QGRNN(node_features, weights):
    # 编码节点特征到指定的量子比特上
    qml.AmplitudeEmbedding(node_features[:8], wires=[0, 1, 2], normalize=True)
    qml.AmplitudeEmbedding(node_features[8:16], wires=[3, 4, 5], normalize=True)
    qml.AmplitudeEmbedding(node_features[16:24], wires=[6, 7, 8], normalize=True)

    # 加入量子层的操作
    for i in range(9):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RZ(weights[i, 2], wires=i)

    # 加入纠缠层（示例：全部CNOT）
    for i in range(8):
        qml.CNOT(wires=[i, i + 1])

    # 测量结果
    return qml.probs(wires=range(9))

class QuantumGraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QuantumGraphConvLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(9, 3))

    def forward(self, node_features):
        # 对每个批次分别计算量子图卷积
        results = [quantum_graph_layer_QGRNN(features, self.weights) for features in node_features]
        return torch.stack(results)


# 测试量子图卷积层
node_features = torch.randn(6, 24)  # 6个样本，每个样本24维特征
quantum_layer = QuantumGraphConvLayer(input_dim=24, output_dim=9)
output = quantum_layer(node_features)
print(output.shape)