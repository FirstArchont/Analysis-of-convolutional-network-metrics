# compare_mnist_models.py
import torch
import torch.nn as nn
import json
import time
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))
from datasets import get_mnist_loaders, get_cifar_loaders
from models import SimpleCNN, CNNWithResidual, FullyConnectedModel, RegularizedResidualCNN
from trainer import train_model
from utils import plot_training_history, count_parameters
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np


FC_CONFIG = {
    "input_size": 784,
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 128},
        {"type": "relu"}
    ]
}


def evaluate_inference_time(model, test_loader, device, num_runs=100):
    model.eval()
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if i >= len(test_loader):
                break
                
            data, _ = next(iter(test_loader))
            data = data.to(device)
            
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return sum(times) / len(times) * 1000


def test_with_MNIST():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 64
    epochs = 5
    learning_rate = 0.001
    
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    
    fc_model = FullyConnectedModel(input_size=784, num_classes=10, layers=FC_CONFIG['layers']).to(device)
    simple_cnn = SimpleCNN(input_channels=1, num_classes=10).to(device)
    residual_cnn = CNNWithResidual(input_channels=1, num_classes=10).to(device)
    
    models = {
        "FullyConnected": fc_model,
        "SimpleCNN": simple_cnn,
        "ResidualCNN": residual_cnn
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}")
        
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
        
        final_train_acc = history['train_accs'][-1]
        final_test_acc = history['test_accs'][-1]
        final_train_loss = history['train_losses'][-1]
        final_test_loss = history['test_losses'][-1]
        
        results[model_name] = {
            "parameters": num_params,
            "training_time_seconds": training_time,
            "inference_time_ms": inference_time,
            "final_train_accuracy": final_train_acc,
            "final_test_accuracy": final_test_acc,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "training_history": history
        }
        
        plot_training_history(history)

    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_MNIST.JSON')
    
    serializable_results = results.copy()
    for model_name in serializable_results:
        serializable_results[model_name].pop("training_history", None)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def get_gradient_flow(model):
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                gradients[name] = {
                    "norm": grad_norm,
                    "mean": grad_mean,
                    "std": grad_std
                }
        return gradients


def get_confusion_matrix_data(model, test_loader, device):
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_predictions)
        class_report = classification_report(all_targets, all_predictions, output_dict=True)
        
        return {
            "confusion_matrix": cm.tolist(),
            "class_report": class_report,
            "predictions": all_predictions,
            "targets": all_targets
        }


def get_overfitting_analysis(train_history):
        train_acc = train_history['train_accs']
        test_acc = train_history['test_accs']
        train_loss = train_history['train_losses']
        test_loss = train_history['test_losses']
        
        overfitting_gap = [train - test for train, test in zip(train_acc, test_acc)]
        loss_gap = [test - train for train, test in zip(train_loss, test_loss)]
        
        return {
            "accuracy_gap": overfitting_gap,
            "loss_gap": loss_gap,
            "final_accuracy_gap": train_acc[-1] - test_acc[-1],
            "final_loss_gap": test_loss[-1] - train_loss[-1],
            "max_accuracy_gap": max(overfitting_gap),
            "max_loss_gap": max(loss_gap)
        }


def test_with_CIFAR():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 64
    epochs = 10
    learning_rate = 0.001
    
    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
    
    fc_model = FullyConnectedModel(input_size=3072, num_classes=10, layers=[
        {"type": "linear", "size": 1024},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2}
    ]).to(device)
    
    residual_cnn = CNNWithResidual(input_channels=3, num_classes=10).to(device)
    
    regularized_cnn = RegularizedResidualCNN(input_channels=3, num_classes=10).to(device)
    
    models = {
        "FullyConnected": fc_model,
        "ResidualCNN": residual_cnn,
        "RegularizedResidualCNN": regularized_cnn
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}")
        
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
        
        gradient_flow = get_gradient_flow(model)
        
        cm_data = get_confusion_matrix_data(model, test_loader, device)
        
        overfitting_analysis = get_overfitting_analysis(history)
        
        results[model_name] = {
            "parameters": num_params,
            "training_time_seconds": training_time,
            "inference_time_ms": inference_time,
            "final_train_accuracy": history['train_accs'][-1],
            "final_test_accuracy": history['test_accs'][-1],
            "final_train_loss": history['train_losses'][-1],
            "final_test_loss": history['test_losses'][-1],
            "training_history": {
                "train_accuracies": history['train_accs'],
                "test_accuracies": history['test_accs'],
                "train_losses": history['train_losses'],
                "test_losses": history['test_losses']
            },
            "overfitting_analysis": overfitting_analysis,
            "confusion_matrix_data": cm_data,
            "gradient_flow": gradient_flow
        }
        
        plot_training_history(history)
        plt.title(f"{model_name} - CIFAR-10 Training History")
        plt.show()
    
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_CIFAR.JSON')
    
    serializable_results = json.loads(json.dumps(results, default=str))
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    return results


# test_with_MNIST()
# test_with_CIFAR()