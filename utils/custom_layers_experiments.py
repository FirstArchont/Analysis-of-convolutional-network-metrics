import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_custom_vs_baseline():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.1.JSON')
    
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    baseline = data['Baseline']['training_history']
    custom = data['Custom']['training_history']
    
    epochs = range(1, len(baseline['train_accuracies']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, baseline['train_accuracies'], 'o-', label='Base (Train)', color='steelblue')
    plt.plot(epochs, baseline['test_accuracies'], 's-', label='Base (Test)', color='steelblue', alpha=0.7)
    plt.plot(epochs, custom['train_accuracies'], 'o-', label='Custom (Train)', color='crimson')
    plt.plot(epochs, custom['test_accuracies'], 's-', label='Custom (Test)', color='crimson', alpha=0.7)
    plt.title('Точность по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, baseline['train_losses'], 'o-', label='Base (Train)', color='steelblue')
    plt.plot(epochs, baseline['test_losses'], 's-', label='Base (Test)', color='steelblue', alpha=0.7)
    plt.plot(epochs, custom['train_losses'], 'o-', label='Custom (Train)', color='crimson')
    plt.plot(epochs, custom['test_losses'], 's-', label='Custom (Test)', color='crimson', alpha=0.7)
    plt.title('Потери по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.suptitle('Сравнение: Стандартные слои (Base) vs Кастомные слои (Custom)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()


def plot_parameters_comparison():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.2.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model_names = ["BasicBlock", "BottleneckBlock", "WideBlock"]
    
    parameters = [data[name]["parameters"] for name in model_names]
    
    colors = ["steelblue", "seagreen", "crimson"]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, parameters, color=colors, alpha=0.8)

    for bar, param in zip(bars, parameters):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(parameters) * 0.01,
            f"{param:,}",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    plt.title("Сравнение количества параметров в Residual блоках", fontsize=14)
    plt.ylabel("Количество параметров", fontsize=12)
    plt.xlabel("Тип Residual блока", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()


def plot_training_stability():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.2.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model_names = ["BasicBlock", "BottleneckBlock", "WideBlock"]
    colors = {
        "BasicBlock": "steelblue",
        "BottleneckBlock": "seagreen",
        "WideBlock": "crimson"
    }

    sample_hist = data[model_names[0]]["training_history"]
    epochs = range(1, len(sample_hist["train_accuracies"]) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for name in model_names:
        hist = data[name]["training_history"]
        gap = [t - v for t, v in zip(hist["train_accuracies"], hist["test_accuracies"])]
        plt.plot(epochs, gap, marker='o', label=name, color=colors[name])
    
    plt.title("Accuracy Gap (Train – Test)")
    plt.xlabel("Эпоха")
    plt.ylabel("Разрыв точности")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)

    plt.subplot(1, 2, 2)
    for name in model_names:
        hist = data[name]["training_history"]
        plt.plot(epochs, hist["test_losses"], marker='s', label=name, color=colors[name])
    
    plt.title("Test Loss по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("Test Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.suptitle("Анализ стабильности обучения Residual блоков", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


def plot_test_accuracy_by_epoch():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.2.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model_names = ["BasicBlock", "BottleneckBlock", "WideBlock"]
    colors = {
        "BasicBlock": "steelblue",
        "BottleneckBlock": "seagreen",
        "WideBlock": "crimson"
    }

    plt.figure(figsize=(10, 6))

    for name in model_names:
        test_accs = data[name]["training_history"]["test_accuracies"]
        epochs = range(1, len(test_accs) + 1)
        plt.plot(epochs, test_accs, marker='o', label=name, color=colors[name], linewidth=2, markersize=5)

    plt.title("Test Accuracy по эпохам для разных Residual блоков", fontsize=14)
    plt.xlabel("Эпоха", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.show()


# plot_custom_vs_baseline()
# plot_test_accuracy_by_epoch()
# plot_parameters_comparison()
# plot_training_stability()
