import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
import time
import json
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))
from models import BaselineModel, CustomModel, WideBlock, BottleneckBlock, BasicBlock, ResNetLike
from datasets import get_cifar_loaders
from utils import count_parameters
from trainer import train_model


def evaluate_inference_time(model, test_loader, device, num_runs=100):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_runs:
                break
            data = data.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    return np.mean(times) * 1000  # ms


def test_custom_layers():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    epochs = 10
    lr = 0.001

    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)

    models = {
        "Baseline": BaselineModel().to(device),
        "Custom": CustomModel().to(device)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name} model")

        num_params = count_parameters(model)

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=str(device)
        )

        inference_time = evaluate_inference_time(model, test_loader, device)

        final_train_acc = history['train_accs'][-1]
        final_test_acc = history['test_accs'][-1]
        final_train_loss = history['train_losses'][-1]
        final_test_loss = history['test_losses'][-1]
        accuracy_gap = final_train_acc - final_test_acc

        results[name] = {
            "parameters": num_params,
            "inference_time_ms": inference_time,
            "final_train_accuracy": final_train_acc,
            "final_test_accuracy": final_test_acc,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "accuracy_gap": accuracy_gap,
            "training_history": {
                "train_accuracies": history['train_accs'],
                "test_accuracies": history['test_accs'],
                "train_losses": history['train_losses'],
                "test_losses": history['test_losses']
            }
        }

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '#3.1.JSON')

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        else:
            return obj

    serializable_results = to_serializable(results)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    return results


def create_models():
    models = {
        "BasicBlock": ResNetLike(BasicBlock, layers=[2, 2, 2], num_classes=10),
        "BottleneckBlock": ResNetLike(BottleneckBlock, layers=[2, 2, 2], num_classes=10),
        "WideBlock": ResNetLike(WideBlock, layers=[2, 2, 2], num_classes=10, width_factor=2)
    }
    return models


def evaluate_inference_time(model, test_loader, device, num_runs=100):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_runs:
                break
            data = data.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    return np.mean(times) * 1000  # ms


def test_residual_blocks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    epochs = 10
    lr = 0.001

    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
    models_dict = create_models()

    results = {}

    for name, model in models_dict.items():
        print(f"Training {name}")

        model = model.to(device)
        num_params = count_parameters(model)

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=str(device)
        )

        inference_time = evaluate_inference_time(model, test_loader, device)

        final_train_acc = history['train_accs'][-1]
        final_test_acc = history['test_accs'][-1]
        final_train_loss = history['train_losses'][-1]
        final_test_loss = history['test_losses'][-1]
        accuracy_gap = final_train_acc - final_test_acc

        results[name] = {
            "parameters": num_params,
            "inference_time_ms": inference_time,
            "final_train_accuracy": final_train_acc,
            "final_test_accuracy": final_test_acc,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "accuracy_gap": accuracy_gap,
            "training_history": {
                "train_accuracies": history['train_accs'],
                "test_accuracies": history['test_accs'],
                "train_losses": history['train_losses'],
                "test_losses": history['test_losses']
            }
        }

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '#3.2.JSON')

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        else:
            return obj

    serializable_results = to_serializable(results)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    return results


# test_custom_layers()
# test_residual_blocks()