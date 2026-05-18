# ============================================================
# MNIST QAT with PyTorch
# 784 -> hidden_size -> hidden_size -> 10
#
# Exports:
#   models/mnist_qat_fp32.onnx
#   models/mnist_qat_fp16.onnx
#   models/mnist_qat_int8.onnx
#   models/accuracy.json
#
# Notes:
# - QAT is done with torch.ao.quantization
# - INT8 is produced by QAT convert()
# - Accuracy for FP32 / FP16 / INT8 is saved in JSON
# ============================================================

import os
import json
import copy
import random
import platform
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quantization

from torch.utils.data import Dataset, DataLoader

import onnxruntime as ort


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Quantization Backend
# ============================================================
def set_quant_backend():
    """
    Must be called before qconfig / prepare_qat / convert.
    """
    machine = platform.machine().lower()

    if "arm" in machine or "aarch" in machine:
        preferred = "qnnpack"
    else:
        preferred = "fbgemm"

    supported = torch.backends.quantized.supported_engines
    print("Supported quantization engines:", supported)

    if preferred not in supported:
        preferred = supported[0]

    torch.backends.quantized.engine = preferred
    print("Using quantization backend:", preferred)


# ============================================================
# Dataset
# ============================================================
class MNISTCSVDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        self.labels = data.iloc[:, 0].values.astype(np.int64)
        self.images = data.iloc[:, 1:].values.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28)
        label = self.labels[idx]

        image = torch.tensor(image).float().unsqueeze(0)  # (1, 28, 28)
        label = torch.tensor(label).long()

        return image, label


def load_mnist_csv_numpy(csv_file):
    data = pd.read_csv(csv_file)
    labels = data.iloc[:, 0].values.astype(np.int64)
    images = data.iloc[:, 1:].values.astype(np.float32) / 255.0
    images = images.reshape(-1, 1, 28, 28)
    return images, labels


# ============================================================
# Model
# ============================================================
class MNISTQATModel(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(784, hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.quant(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.dequant(x)
        return x


# ============================================================
# Fuse
# ============================================================
def fuse_model(model):
    """
    Fuse Linear + ReLU for QAT.
    Must be called when model is in eval mode.
    """
    quantization.fuse_modules(
        model,
        [
            ["fc1", "relu1"],
            ["fc2", "relu2"],
        ],
        inplace=True,
    )


# ============================================================
# PyTorch Evaluation
# ============================================================
def evaluate_pytorch_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ============================================================
# ONNX Evaluation
# ============================================================
def evaluate_onnx_model(onnx_path, csv_file):
    images, labels = load_mnist_csv_numpy(csv_file)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type

    correct = 0
    total = len(labels)

    use_fp16 = "float16" in input_type.lower()

    for i in range(total):
        x = images[i : i + 1]
        if use_fp16:
            x = x.astype(np.float16)
        else:
            x = x.astype(np.float32)

        outputs = session.run(None, {input_name: x})
        pred = np.argmax(outputs[0], axis=1)[0]

        if pred == labels[i]:
            correct += 1

    return correct / total


# ============================================================
# Train FP32 and QAT
# ============================================================
def train_qat_model(
    train_csv="mnist_train.csv",
    test_csv="mnist_test.csv",
    hidden_size=128,
    batch_size=128,
    epochs_fp32=5,
    epochs_qat=10,
    learning_rate=0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MNISTCSVDataset(train_csv)
    test_dataset = MNISTCSVDataset(test_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # FP32 training
    # --------------------------------------------------------
    model_fp32 = MNISTQATModel(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model_fp32.parameters(), lr=learning_rate)

    print("\n========== FP32 PRETRAIN ==========")

    for epoch in range(epochs_fp32):
        model_fp32.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model_fp32(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate_pytorch_model(model_fp32, test_loader, device)
        print(
            f"[FP32] Epoch {epoch+1}/{epochs_fp32} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Acc: {acc:.4f}"
        )

    # --------------------------------------------------------
    # QAT preparation
    # IMPORTANT: prepare_qat requires model.train() == True
    # --------------------------------------------------------
    qat_model = copy.deepcopy(model_fp32).cpu()

    qat_model.eval()  # required before fuse
    fuse_model(qat_model)

    qat_model.train()  # required before prepare_qat
    qat_model.qconfig = quantization.get_default_qat_qconfig(
        torch.backends.quantized.engine
    )

    quantization.prepare_qat(qat_model, inplace=True)

    qat_model = qat_model.to(device)
    optimizer_qat = optim.Adam(qat_model.parameters(), lr=learning_rate * 0.1)

    print("\n========== QAT FINE-TUNING ==========")

    for epoch in range(epochs_qat):
        qat_model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer_qat.zero_grad()
            outputs = qat_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_qat.step()

            total_loss += loss.item()

        acc = evaluate_pytorch_model(qat_model, test_loader, device)
        print(
            f"[QAT] Epoch {epoch+1}/{epochs_qat} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Acc: {acc:.4f}"
        )

    # --------------------------------------------------------
    # Convert QAT model to true INT8
    # --------------------------------------------------------
    qat_model = qat_model.cpu()
    qat_model.eval()

    int8_model = copy.deepcopy(qat_model)
    quantization.convert(int8_model, inplace=True)

    int8_acc = evaluate_pytorch_model(int8_model, test_loader, torch.device("cpu"))

    return model_fp32.cpu(), int8_model, test_loader, test_csv


# ============================================================
# Export ONNX files
# ============================================================
def export_onnx_models(fp32_model, int8_model):
    os.makedirs("models", exist_ok=True)

    dummy_input_fp32 = torch.randn(1, 1, 28, 28)
    dummy_input_fp16 = dummy_input_fp32.half()

    # FP32 ONNX
    fp32_model = fp32_model.eval().cpu()
    torch.onnx.export(
        fp32_model,
        dummy_input_fp32,
        "models/mnist_qat_fp32.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    print("Saved: models/mnist_qat_fp32.onnx")

    # FP16 ONNX
    fp16_model = copy.deepcopy(fp32_model).half()
    torch.onnx.export(
        fp16_model,
        dummy_input_fp16,
        "models/mnist_qat_fp16.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    print("Saved: models/mnist_qat_fp16.onnx")

    # INT8 ONNX
    int8_export_ok = True
    try:
        int8_model = int8_model.eval().cpu()
        torch.onnx.export(
            int8_model,
            dummy_input_fp32,
            "models/mnist_qat_int8.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
        )
        print("Saved: models/mnist_qat_int8.onnx")
    except Exception as e:
        int8_export_ok = False
        print("INT8 ONNX export failed:")
        print(e)

    return int8_export_ok


# ============================================================
# Accuracy JSON
# ============================================================
def save_accuracy_json(fp32_acc, fp16_acc, int8_acc, int8_export_ok):
    os.makedirs("models", exist_ok=True)

    accuracy_data = {
        "fp32_accuracy": round(float(fp32_acc), 6),
        "fp16_accuracy": round(float(fp16_acc), 6),
        "int8_accuracy": round(float(int8_acc), 6),
        "int8_onnx_exported": bool(int8_export_ok),
        "note": "Accuracies are computed from exported ONNX files when available.",
    }

    with open("models/accuracy.json", "w") as f:
        json.dump(accuracy_data, f, indent=4)

    print("Saved: models/accuracy.json")


# ============================================================
# Main
# ============================================================
def main():
    set_seed(42)
    set_quant_backend()

    fp32_model, int8_model, test_loader, test_csv = train_qat_model(
        train_csv="mnist_train.csv",
        test_csv="mnist_test.csv",
        hidden_size=128,
        batch_size=128,
        epochs_fp32=5,
        epochs_qat=10,
        learning_rate=0.001,
    )

    int8_export_ok = export_onnx_models(fp32_model, int8_model)

    print("\n========== FINAL ACCURACY SUMMARY ==========")

    fp32_acc = evaluate_onnx_model("models/mnist_qat_fp32.onnx", test_csv)
    fp16_acc = evaluate_onnx_model("models/mnist_qat_fp16.onnx", test_csv)

    if int8_export_ok:
        int8_acc = evaluate_onnx_model("models/mnist_qat_int8.onnx", test_csv)
    else:
        int8_acc = evaluate_pytorch_model(int8_model, test_loader, torch.device("cpu"))

    print(f"FP32 Accuracy : {fp32_acc:.4f}")
    print(f"FP16 Accuracy : {fp16_acc:.4f}")
    print(f"INT8 Accuracy : {int8_acc:.4f}")
    print("===========================================")

    save_accuracy_json(fp32_acc, fp16_acc, int8_acc, int8_export_ok)


if __name__ == "__main__":
    main()
