import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

# 量子比特数量
qbits3 = 8

# Pennylane设备
dev = qml.device("default.mixed", wires=qbits3)


def conv(b1, b2, params):
    qml.RZ(-torch.pi / 2, wires=b2)
    qml.CNOT([b2, b1])
    qml.RZ(params[0], wires=b1)
    qml.RY(params[1], wires=b2)
    qml.CNOT([b1, b2])
    qml.RY(params[2], wires=b2)
    qml.CNOT([b2, b1])
    qml.RZ(torch.pi / 2, wires=b1)


def pool(b1, b2, params):
    qml.RZ(-torch.pi / 2, wires=b2)
    qml.CNOT([b2, b1])
    qml.RZ(params[0], wires=b1)
    qml.RY(params[1], wires=b2)
    qml.CNOT([b1, b2])
    qml.RY(params[2], wires=b2)


def swap(b1, b2):
    qml.SWAP(wires=[b1, b2])


# 添加噪声的量子图卷积层
@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_fusion_layer(node_features, weights, coupling, noise_strength):
    qml.AmplitudeEmbedding(node_features, wires=range(qbits3), normalize=True)

    # 单量子比特旋转
    for i in range(qbits3):
        qml.RX(weights[i], wires=i)
        qml.RY(weights[i + qbits3], wires=i)
        qml.RZ(weights[i + 2 * qbits3], wires=i)

        # 添加BF噪声
        qml.BitFlip(noise_strength, wires=i)

        # 添加 PF 噪声
        #qml.PhaseFlip(noise_strength, wires=i)

        # 添加AD噪声
        #qml.AmplitudeDamping(noise_strength, wires=i)

        # 添加去相干噪声DN
        #qml.DepolarizingChannel(noise_strength, wires=i)

    for i in range(qbits3):
        for j in range(i + 1, qbits3):
            qml.IsingXX(coupling[i, j], wires=[i, j])


    # 量子卷积和池化操作
    coupling_min = coupling.min()
    coupling_max = coupling.max()
    coupling_normalized = (coupling - coupling_min) / (coupling_max - coupling_min)
    adj_matrix = (coupling_normalized > 0.5).float()

    weight_idx = 3 * qbits3
    for i in range(qbits3):
        for j in range(i + 1, qbits3):
            if adj_matrix[i, j] == 1:
                #if abs(i - j) != 1:
                    #swap(i, j)
                conv(i, j, weights[weight_idx:weight_idx + 3])
                pool(i, j, weights[weight_idx + 3:weight_idx + 6])

                # 添加BF噪声
                #qml.BitFlip(noise_strength, wires=i)

                #添加 PF 噪声
                #qml.PhaseFlip(noise_strength, wires=i)

                # 添加AD噪声
                #qml.AmplitudeDamping(noise_strength, wires=i)

                # 添加去相干噪声DN
                #qml.DepolarizingChannel(noise_strength, wires=i)


                weight_idx += 6
                #if abs(i - j) != 1:
                    #swap(i, j)

    return qml.state()


# 初始化权重和耦合常数
node_features = np.random.rand(256)
weights = np.random.rand(3 * qbits3 * (qbits3 - 1) // 2 + 6 * (qbits3 - 1) * (qbits3 - 2) // 2)
coupling = np.random.rand(qbits3, qbits3)

# 量子图卷积层类
class QMFNN(nn.Module):
    def __init__(self, n_qubits, weights, coupling):
        super(QMFNN, self).__init__()
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        self.coupling = nn.Parameter(torch.tensor(coupling, dtype=torch.float32))

    def forward(self, x):
        q_out = quantum_fusion_layer(x, self.weights, self.coupling, 0)
        noisy_state = quantum_fusion_layer(x, self.weights, self.coupling, 0.01)
        original_state=q_out
        # 计算保真度
        fidelity = qml.math.fidelity(original_state, noisy_state)

        # 确保保真度在 [0, 1] 范围内

        print(f"Fidelity: {fidelity}")



        return q_out


# 测试抗噪声性能
model = QMFNN(qbits3, weights, coupling)
input_features = torch.tensor(node_features, dtype=torch.float32)
output = model(input_features)

#print("Output with noise:", output)


#Fidelity: 0.9932339216941617