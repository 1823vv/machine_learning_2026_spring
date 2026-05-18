"""Train a small MNIST classifier, export ONNX, and create an INT8 model.

This script is designed for teaching quantization, not for winning MNIST.
It uses the CSV files in this folder, trains a compact scikit-learn MLP,
exports the model to ONNX, dynamically quantizes the ONNX graph, and writes
simple metrics for comparison.

Install dependencies from this folder:
    pip install -r requirements.txt

Optional graph viewer:
    netron is included in requirements.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_mnist_csv(path: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load an MNIST CSV file with a label column followed by 784 pixels."""
    frame = pd.read_csv(path, nrows=limit)
    y = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    x = frame.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
    return x, y


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Train a small model that exports cleanly to ONNX for inspection."""
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    solver="adam",
                    max_iter=12,
                    batch_size=128,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def export_onnx(model: Pipeline, output_path: Path) -> None:
    """Export the scikit-learn pipeline to ONNX."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("float_input", FloatTensorType([None, 784]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=15)
    output_path.write_bytes(onnx_model.SerializeToString())


def quantize_dynamic(fp32_path: Path, int8_path: Path) -> None:
    """Create a dynamically quantized ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )


def evaluate_onnx(model_path: Path, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate an ONNX model with ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x_test.astype(np.float32)})

    # skl2onnx classifiers usually return labels as the first output.
    predicted = outputs[0]
    if predicted.ndim > 1:
        predicted = np.argmax(predicted, axis=1)

    return float(accuracy_score(y_test, predicted))


def file_size(path: Path) -> int:
    """Return file size in bytes."""
    return path.stat().st_size


def main() -> None:
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)

    train_csv = root / "mnist_train.csv"
    test_csv = root / "mnist_test.csv"

    # Limits keep the teaching script quick on ordinary laptops.
    x_train, y_train = load_mnist_csv(train_csv, limit=8000)
    x_test, y_test = load_mnist_csv(test_csv, limit=2000)

    model = train_model(x_train, y_train)
    sklearn_accuracy = float(model.score(x_test, y_test))

    fp32_path = artifacts / "mnist_mlp_fp32.onnx"
    int8_path = artifacts / "mnist_mlp_dynamic_int8.onnx"

    export_onnx(model, fp32_path)
    quantize_dynamic(fp32_path, int8_path)

    metrics: dict[str, Any] = {
        "sklearn_accuracy": sklearn_accuracy,
        "fp32_onnx": {
            "path": str(fp32_path.relative_to(root)),
            "bytes": file_size(fp32_path),
            "accuracy": evaluate_onnx(fp32_path, x_test, y_test),
        },
        "dynamic_int8_onnx": {
            "path": str(int8_path.relative_to(root)),
            "bytes": file_size(int8_path),
            "accuracy": evaluate_onnx(int8_path, x_test, y_test),
        },
    }

    metrics["size_reduction_ratio"] = round(
        metrics["fp32_onnx"]["bytes"] / metrics["dynamic_int8_onnx"]["bytes"],
        3,
    )

    metrics_path = artifacts / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("\nOpen these files in Netron:")
    print(f"  {fp32_path}")
    print(f"  {int8_path}")


if __name__ == "__main__":
    main()
