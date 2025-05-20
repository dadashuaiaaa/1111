import pennylane as qml
from pennylane import numpy as np

# 量子比特数量
qbits3 = 8

# 定义量子设备
dev = qml.device("default.mixed", wires=qbits3)

# 定义量子图卷积层
@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_fusion_layer(node_features, weights, coupling, noise_strength):

    qml.AmplitudeEmbedding(node_features, wires=range(qbits3), normalize=True)

    # 单量子比特旋转
    for i in range(qbits3):
        qml.RX(weights[i], wires=i)
        qml.RY(weights[i + qbits3], wires=i)
        qml.RZ(weights[i + 2 * qbits3], wires=i)


    # 施加噪声（每个量子比特上应用去相干噪声）
    #for i in range(qbits3):
        #qml.DepolarizingChannel(noise_strength, wires=i)

    # 应用耦合操作
    for i in range(qbits3):
        for j in range(i + 1, qbits3):
            qml.IsingXX(coupling[i, j], wires=[i, j])


    # 返回密度矩阵
    return qml.state()


# 初始化权重和耦合常数
node_features = np.random.rand(256)
weights = np.random.rand(3 * qbits3)
coupling = np.random.rand(qbits3, qbits3)
#noise_strength = 0.01 # 噪声强度

# 获取原始量子态
original_state = quantum_fusion_layer(node_features, weights, coupling, 0)

# 为了对比，施加相同的噪声
noisy_state = quantum_fusion_layer(node_features, weights, coupling, 0.01)

# 计算保真度
fidelity = qml.math.fidelity(original_state, noisy_state)

# 确保保真度在 [0, 1] 范围内

print(f"Fidelity: {fidelity}")

