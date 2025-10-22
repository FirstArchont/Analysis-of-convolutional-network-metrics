import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))
from models import Conv3x3CNN, Conv5x5CNN, Conv7x7CNN, MixedKernelCNN, ShallowCNN, MediumCNN, DeepCNN, ResidualCNN
from datasets import get_cifar_loaders
from trainer import train_model
from utils import count_parameters
import matplotlib.pyplot as plt


def get_receptive_field(kernel_size, stride, padding, current_rf, current_stride):
    rf = current_rf + (kernel_size - 1) * current_stride
    new_stride = current_stride * stride
    return rf, new_stride


def calculate_receptive_field(model, input_size=32):
    rf, stride = 1, 1
    for i in range(3):
        if hasattr(model, 'conv1'):
            kernel_size = model.conv1.kernel_size[0] if isinstance(model.conv1.kernel_size, tuple) else model.conv1.kernel_size
            rf, stride = get_receptive_field(kernel_size, 1, 1, rf, stride)
        rf, stride = get_receptive_field(2, 2, 0, rf, stride)  # MaxPool
    
    return rf


def get_first_layer_activations(model_name, model, test_loader, device, num_images=5):
    model.eval()
    activations = []
    
    hook_handle = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    if hasattr(model, 'conv1'):
        hook = model.conv1.register_forward_hook(hook_fn)
        hook_handle.append(hook)
    elif hasattr(model, 'conv1_1x1'):
        hook = model.conv1_1x1.register_forward_hook(hook_fn)
        hook_handle.append(hook)
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 1:
                break
            data = data.to(device)
            _ = model(data)
    
    for handle in hook_handle:
        handle.remove()

    if activations:
        plt.imshow(activations[0][0, 0], cmap='hot', interpolation='nearest')
        plt.title(f"{model_name} First Layer Activation")
        plt.show()
    
    return activations[0] if activations else None


def evaluate_inference_time(model, test_loader, device, num_runs=50):
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_runs:
                break
            data = data.to(device)
            
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return sum(times) / len(times) * 1000


def test_kernel_sizes():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 64
    epochs = 10
    learning_rate = 0.001
    
    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
    
    models = {
        "Conv3x3": Conv3x3CNN().to(device),
        "Conv5x5": Conv5x5CNN().to(device),
        "Conv7x7": Conv7x7CNN().to(device),
        "Mixed1x1_3x3": MixedKernelCNN().to(device)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}")
        
        num_params = count_parameters(model)
        
        receptive_field = calculate_receptive_field(model)
        
        start_train_time = time.time()
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=learning_rate,
            device=str(device)
        )
        end_train_time = time.time()
        training_time = end_train_time - start_train_time
        
        inference_time = evaluate_inference_time(model, test_loader, device)
        
        first_layer_activations = get_first_layer_activations(model_name, model, test_loader, device)
        
        final_train_acc = history['train_accs'][-1]
        final_test_acc = history['test_accs'][-1]
        final_train_loss = history['train_losses'][-1]
        final_test_loss = history['test_losses'][-1]
        
        accuracy_gap = final_train_acc - final_test_acc
        loss_gap = final_test_loss - final_train_loss
        
        results[model_name] = {
            "parameters": num_params,
            "training_time_seconds": training_time,
            "inference_time_ms": inference_time,
            "final_train_accuracy": final_train_acc,
            "final_test_accuracy": final_test_acc,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "accuracy_gap": accuracy_gap,
            "loss_gap": loss_gap,
            "receptive_field": receptive_field,
            "training_history": {
                "train_accuracies": history['train_accs'],
                "test_accuracies": history['test_accs'],
                "train_losses": history['train_losses'],
                "test_losses": history['test_losses']
            },
        }

        if first_layer_activations is not None:
            sample_activation = first_layer_activations[0, 0]
            results[model_name]["first_layer_activation_sample"] = sample_activation.tolist()
        else:
            results[model_name]["first_layer_activation_sample"] = None
        
        if first_layer_activations is not None:
            print(f"Активации {model_name}: min={first_layer_activations.min():.4f}, max={first_layer_activations.max():.4f}, mean={first_layer_activations.mean():.4f}")
    
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.1.JSON')
    serializable_results = json.loads(json.dumps(results, default=str))
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    return results


def compute_gradient_norms(model, train_loader, device, loss_fn=nn.CrossEntropyLoss()):
    model.train()
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.abs().mean().item()
        else:
            grad_norms[name] = 0.0

    model.zero_grad()
    return grad_norms


def test_cnn_depth():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)

    models = {
        "ShallowCNN": ShallowCNN().to(device),
        "MediumCNN": MediumCNN().to(device),
        "DeepCNN": DeepCNN().to(device),
        "ResidualCNN": ResidualCNN().to(device)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        num_params = count_parameters(model)

        start_train_time = time.time()
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=learning_rate,
            device=str(device)
        )
        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        inference_time = evaluate_inference_time(model, test_loader, device)

        first_layer_activations = get_first_layer_activations(model, test_loader, device, num_images=5)

        grad_norms = compute_gradient_norms(model, train_loader, device)

        final_train_acc = history['train_accs'][-1]
        final_test_acc = history['test_accs'][-1]
        final_train_loss = history['train_losses'][-1]
        final_test_loss = history['test_losses'][-1]
        accuracy_gap = final_train_acc - final_test_acc
        loss_gap = final_test_loss - final_train_loss

        results[model_name] = {
            "parameters": num_params,
            "training_time_seconds": training_time,
            "inference_time_ms": inference_time,
            "final_train_accuracy": final_train_acc,
            "final_test_accuracy": final_test_acc,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "accuracy_gap": accuracy_gap,
            "loss_gap": loss_gap,
            "receptive_field": calculate_receptive_field(model),
            "gradient_norms_by_layer": grad_norms,
            "training_history": {
                "train_accuracies": history['train_accs'],
                "test_accuracies": history['test_accs'],
                "train_losses": history['train_losses'],
                "test_losses": history['test_losses']
            },
            "first_layer_activations_shape": first_layer_activations.shape if first_layer_activations is not None else None,
            "first_layer_activations_sample": first_layer_activations[0].tolist() if first_layer_activations is not None else None
        }

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '#2.2.JSON')

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj

    serializable_results = make_serializable(results)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    return results


test_kernel_sizes()
# test_cnn_depth()