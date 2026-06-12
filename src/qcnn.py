import pandas as pd
import numpy as np
import os
import re
from Bio import SeqIO
from PyBioMed.PyDNA.PyDNAac import GetDAC
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap as z_feature_map
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from sklearn.metrics import classification_report, confusion_matrix
import json

from src import tools
from src import preprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir)

ALGORITHM_NAME = "QCNN_Hybrid_Weighted"
output_folder = f"{ALGORITHM_NAME}-result"

data_dir = os.path.join(PROJECT_ROOT, 'data')
result_dir = os.path.join(PROJECT_ROOT, output_folder)

algorithm_globals.random_seed = 12345
estimator = Estimator()

os.makedirs(result_dir, exist_ok=True)

def train():
    df, df_seq, x, y, y_map = preprocess.run()

    # We now define a two qubit unitary as defined in [3]
    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target


    # Let's draw this circuit and see what it looks like
    params = ParameterVector("θ", length=3)
    circuit = conv_circuit(params)
    circuit.draw("mpl", style="clifford", filename=os.path.join(data_dir, "conv_circuit.jpg"))

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc


    circuit = conv_layer(4, "θ")
    circuit.decompose().draw("mpl", style="clifford", filename=os.path.join(data_dir, "conv_layer.jpg"))

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)

        return target


    params = ParameterVector("θ", length=3)
    circuit = pool_circuit(params)
    circuit.draw("mpl", style="clifford", filename=os.path.join(data_dir, "pool_circuit.jpg"))

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc


    sources = [0, 1]
    sinks = [2, 3]
    circuit = pool_layer(sources, sinks, "θ")
    circuit.decompose().draw("mpl", style="clifford", filename=os.path.join(data_dir, "pool_layer.jpg"))

    # ==========================================
    # 1. DEFINISI ARSITEKTUR SIKUIT QCNN (QISKIT)
    # ==========================================
    feature_map = z_feature_map(12)
    ansatz = QuantumCircuit(12, name="Ansatz")

    # LAYER 1 (12 Qubit -> Sisa 6 Qubit)
    ansatz.compose(conv_layer(12, "c1"), list(range(12)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], "p1"), list(range(12)), inplace=True)

    # LAYER 2 (6 Qubit aktif -> Sisa 3 Qubit)
    ansatz.compose(conv_layer(6, "c2"), list(range(6, 12)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2], [3, 4, 5], "p2"), list(range(6, 12)), inplace=True)

    # PENGGABUNGAN SIKUIT
    circuit = QuantumCircuit(12)
    circuit.compose(feature_map, range(12), inplace=True)
    circuit.compose(ansatz, range(12), inplace=True)

    circuit.draw("mpl", style="clifford", filename=os.path.join(data_dir, "qcnn.jpg"))

    # Observables untuk mengukur 3 qubit aktif akhir (indeks 9, 10, 11)
    observables = [
        SparsePauliOp.from_list([("I" * 9 + "Z" + "I" * 2, 1)]),  
        SparsePauliOp.from_list([("I" * 10 + "Z" + "I" * 1, 1)]), 
        SparsePauliOp.from_list([("I" * 11 + "Z", 1)])           
    ]

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

    # ==========================================
    # 2. PERHITUNGAN WEIGHT KELAS DINAMIS (PYTORCH)
    # ==========================================
    # Menggunakan baris kode perhitungan bobot yang Anda buat
    num_label = df_seq['Resistance Mechanism'].value_counts().tolist()
    total_label = len(num_label)
    class_weights = [sum(num_label) / (total_label * num_label[i]) for i in range(len(num_label))]
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Definisikan loss function dengan menyertakan bobot dinamis tersebut
    criterion = nn.CrossEntropyLoss(weight=class_weights)


    # ==========================================
    # 3. PEMBUATAN MODEL HIBRIDA QML & LINEAR LAYER
    # ==========================================
    class HybridQCNNClassifier(nn.Module):
        def __init__(self, qiskit_qnn, num_classes):
            super().__init__()
            # TorchConnector mengubah QNN Qiskit menjadi module PyTorch yang autograd-compatible
            self.quantum_layer = TorchConnector(qiskit_qnn)
            
            # Jembatan dimensi: memetakan 3 ekspektasi nilai qubit menjadi jumlah kelas (5 kelas CARD)
            self.classical_layer = nn.Linear(3, num_classes)
            
        def forward(self, x):
            # x memiliki bentuk input batch (batch_size, 12)
            q_out = self.quantum_layer(x)       # Hasil pengukuran kuantum berukuran (batch_size, 3)
            logits = self.classical_layer(q_out) # Logits kelas berukuran (batch_size, 5)
            return logits

    # Inisialisasi model hibrida dengan total_label (5 kelas)
    model = HybridQCNNClassifier(qiskit_qnn=qnn, num_classes=total_label)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # x and y to tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)

    model.train()
    optimizer.zero_grad()

    outputs = model(x_train)       # Prediksi dari model hibrida
    loss = criterion(outputs, y_train) # Evaluasi menggunakan Weighted Cross-Entropy

    loss.backward()                # Backpropagation menembus layer klasik dan sirkuit kuantum sekaligus
    optimizer.step()               # Memperbarui parameter gerbang ansatz dan bobot linear layer

    print(f"Nilai Loss dengan Cost-Sensitive Learning: {loss.item()}")

    # Define algorithm name for the output folder
    ALGORITHM_NAME = "QCNN_Hybrid_Weighted"
    output_folder = f"{ALGORITHM_NAME}-result"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    else:
        print(f"Output directory already exists: {output_folder}")

    # 1. Save the trained PyTorch model's state dictionary
    model_path = os.path.join(output_folder, "hybrid_qcnn_model_state.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model state to: {model_path}")

    # 2. Save the class weights
    class_weights_path = os.path.join(output_folder, "class_weights.pt")
    torch.save(class_weights, class_weights_path)
    print(f"Saved class weights to: {class_weights_path}")

    # 3. Save the y_map (mapping from class name to index)
    ymap_path = os.path.join(output_folder, "y_map.json")
    # Invert y_map to map indices back to class names for better readability in evaluation reports
    inverted_y_map = {v: k for k, v in y_map.items()}
    with open(ymap_path, "w") as f:
        json.dump(inverted_y_map, f, indent=4)
    print(f"Saved y_map to: {ymap_path}")

    # 4. Save the combined Quantum Circuit (feature_map + ansatz) as QASM
    # circuit_qasm_path = os.path.join(output_folder, "hybrid_qcnn_circuit.qasm")
    # with open(circuit_qasm_path, "w") as f:
    #    f.write(circuit.qasm())
    # print(f"Saved quantum circuit to: {circuit_qasm_path}")

    # 5. Perform evaluation and save metrics
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for evaluation
        test_outputs = model(x_test)
        _, predicted_labels = torch.max(test_outputs, 1)

    # Convert tensors to numpy arrays for sklearn metrics
    y_test_np = y_test.numpy()
    predicted_labels_np = predicted_labels.numpy()

    # Generate target names from the inverted y_map for the classification report
    target_names = [inverted_y_map[i] for i in sorted(inverted_y_map.keys())]

    # Calculate classification report
    report = classification_report(y_test_np, predicted_labels_np, target_names=target_names, output_dict=True)
    report_path = os.path.join(output_folder, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved classification report to: {report_path}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test_np, predicted_labels_np)
    cm_path = os.path.join(output_folder, "confusion_matrix.npy")
    np.save(cm_path, cm)
    print(f"Saved confusion matrix to: {cm_path}")

    # Print summary
    print("\nEvaluation Results:")
    print(classification_report(y_test_np, predicted_labels_np, target_names=target_names))

if __name__ == "__main__":
    # Ini agar skrip tetap bisa dijalankan langsung lewat terminal
    train()
