import pennylane as qml
import torch

# 量子比特数量
qbits3 = 8

# Pennylane设备
dev = qml.device("default.qubit", wires=qbits3)


# 定义量子图卷积层
@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_fusion_layer(node_features, weights, adj_matrix):
    # 将输入的特征向量嵌入到量子态中
    qml.AmplitudeEmbedding(node_features, wires=range(qbits3), normalize=True)

    # 单量子比特旋转
    for i in range(qbits3):
        qml.RX(weights[i], wires=i)
        qml.RY(weights[i + qbits3], wires=i)
        qml.RZ(weights[i + 2 * qbits3], wires=i)

    # 根据邻接矩阵应用 IsingXX 门
    for i in range(qbits3):
        for j in range(i + 1, qbits3):
            if adj_matrix[i, j] == 1:  # 如果邻接矩阵中对应位置有连边
                qml.IsingXX(1.0, wires=[i, j])  # 可以根据需要调整强度

    # 返回每个量子比特的测量值
    return [qml.expval(qml.PauliZ(i)) for i in range(qbits3)]


# 初始化权重和耦合常数矩阵
weights = torch.randn(3 * qbits3, requires_grad=True)

# 初始化耦合常数矩阵（随机生成的矩阵）
coupling = torch.randn((qbits3, qbits3), requires_grad=False)

# 保证耦合矩阵是对称的
coupling = (coupling + coupling.T) / 2

# 对耦合矩阵进行标准化操作，使其数值在 [0, 1] 范围内
coupling_min = coupling.min()
coupling_max = coupling.max()
coupling_normalized = (coupling - coupling_min) / (coupling_max - coupling_min)

# 根据阈值生成邻接矩阵，阈值设为 0.5
adj_matrix = (coupling_normalized > 0.5).float()
print(adj_matrix)
# 初始化量子比特的输入特征向量，长度必须为 2^qbits3
node_features = torch.randn(2 ** qbits3)
node_features = node_features / torch.norm(node_features)  # 归一化特征

# 调用量子图卷积层
output = quantum_fusion_layer(node_features, weights, adj_matrix)

# 查看输出
print(output)
