```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# --------------------------
# Dataset
# --------------------------
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
        image = torch.tensor(image).float().unsqueeze(0)
        label = torch.tensor(label).long()
        return image, label

# --------------------------
# Model
# --------------------------
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 10)
        )

    def forward(self, x):
        return self.network(x)

# --------------------------
# Training function
# --------------------------
def train_model():
    train_dataset = MNISTCSVDataset("mnist_train.csv")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = MNISTModel()
    model = model.float()  # Ensure FP32

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Loss {total_loss:.4f}")

    return model

# --------------------------
# Export to ONNX
# --------------------------
def export_onnx(model):
    os.makedirs("models", exist_ok=True)
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        "models/mnist_model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )
    print("ONNX model exported to models/mnist_model.onnx")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    model = train_model()
    export_onnx(model)
```

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16
import os

os.makedirs("models", exist_ok=True)

fp32_model = "models/mnist_model.onnx"

# --------------------------
# FP16 Quantization
# --------------------------
model = onnx.load(fp32_model)
fp16_model = float16.convert_float_to_float16(model)
onnx.save(fp16_model, "models/mnist_model_quantized_fp16.onnx")
print("FP16 model saved.")

# --------------------------
# INT8 Quantization
# --------------------------
quantize_dynamic(
    fp32_model,
    "models/mnist_model_quantized_int8.onnx",
    weight_type=QuantType.QInt8
)
print("INT8 model saved.")
```

```python
import onnxruntime as ort
import pandas as pd
import numpy as np
import glob
import os


# --------------------------
# Load MNIST Test Data
# --------------------------
def load_dataset():
    data = pd.read_csv("mnist_test.csv")
    labels = data.iloc[:, 0].values
    images = data.iloc[:, 1:].values.astype(np.float32) / 255.0
    images = images.reshape(-1, 1, 28, 28)
    return images, labels


images, labels = load_dataset()

# --------------------------
# Evaluate all ONNX models in models/ folder
# --------------------------
onnx_files = glob.glob("models/*.onnx")

for model_file in onnx_files:
    session = ort.InferenceSession(model_file)
    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type

    correct = 0

    for i in range(len(images)):
        img = images[i : i + 1]

        if "float16" in input_type:
            img = img.astype(np.float16)

        outputs = session.run(None, {input_name: img})
        pred = np.argmax(outputs[0])
        if pred == labels[i]:
            correct += 1

    accuracy = correct / len(images)
    print(f"{os.path.basename(model_file)} Accuracy: {accuracy:.4f}")
```

```python
import onnx

# Check model information
def check_model_info(model_path):
    model = onnx.load(model_path)
    print(f"\nModel: {model_path}")
    print(f"Number of nodes: {len(model.graph.node)}")

    # Check initializer (weight) data types
    for tensor in model.graph.initializer:
        if tensor.data_type == 1:  # FLOAT
            print(f"  Weight: {tensor.name} - FLOAT")
        elif tensor.data_type == 10:  # FLOAT16
            print(f"  Weight: {tensor.name} - FLOAT16")
        elif tensor.data_type == 3:  # INT8
            print(f"  Weight: {tensor.name} - INT8")


# Check all models
for model_file in onnx_files:
    check_model_info(model_file)
```

```python
def compare_outputs(model1_path, model2_path, num_samples=10):
    # Load the two models
    session1 = ort.InferenceSession(model1_path)
    session2 = ort.InferenceSession(model2_path)

    input_name1 = session1.get_inputs()[0].name
    input_name2 = session2.get_inputs()[0].name

    # Prepare identical random inputs
    np.random.seed(42)
    dummy_input = np.random.randn(num_samples, 1, 28, 28).astype(np.float32)

    # Convert types according to model input (e.g., float16 models)
    if "float16" in session1.get_inputs()[0].type:
        input1 = dummy_input.astype(np.float16)
    else:
        input1 = dummy_input.astype(np.float32)

    if "float16" in session2.get_inputs()[0].type:
        input2 = dummy_input.astype(np.float16)
    else:
        input2 = dummy_input.astype(np.float32)

    # Get outputs
    outputs1 = []
    outputs2 = []

    for i in range(num_samples):
        out1 = session1.run(None, {input_name1: input1[i : i + 1]})
        out2 = session2.run(None, {input_name2: input2[i : i + 1]})
        outputs1.append(out1[0])
        outputs2.append(out2[0])

    # Compare output differences
    total_diff = 0
    for i in range(num_samples):
        diff = np.abs(outputs1[i] - outputs2[i]).mean()
        total_diff += diff

    avg_diff = total_diff / num_samples
    return avg_diff


# Compare original and quantized models
print(
    "Original vs FP16:",
    compare_outputs(
        "models/mnist_model.onnx", "models/mnist_model_quantized_fp16.onnx"
    ),
)
print(
    "Original vs INT8:",
    compare_outputs(
        "models/mnist_model.onnx", "models/mnist_model_quantized_int8.onnx"
    ),
)
```
