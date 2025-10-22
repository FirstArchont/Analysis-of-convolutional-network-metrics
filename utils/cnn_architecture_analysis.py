import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))
from models import ShallowCNN, MediumCNN, DeepCNN, ResidualCNN


def plot_accuracy_vs_training_time():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.1.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = list(data.keys())
    test_accuracies = [data[model]['final_test_accuracy'] for model in models]
    training_times = [data[model]['training_time_seconds'] for model in models]
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, test_accuracies, width, label='Test Accuracy', 
                   color='lightblue', edgecolor='blue', linewidth=2, alpha=0.8)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, training_times, width, label='Время обучения (секунды)', 
                   color='lightcoral', edgecolor='red', linewidth=2, alpha=0.8)

    ax1.set_xlabel('Модели', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold', color='blue')
    ax2.set_ylabel('Время обучения (секунды)', fontsize=14, fontweight='bold', color='red')
    ax1.set_title('Сравнение Test Accuracy и времени обучения',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    
    ax1.set_ylim(0.65, 0.8)
    ax2.set_ylim(150, 175)
    
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', color='blue', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', color='red', fontsize=10)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_receptive_field_analysis():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.1.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = list(data.keys())
    receptive_fields = [data[model]['receptive_field'] for model in models]
    test_accuracies = [data[model]['final_test_accuracy'] for model in models]
    accuracy_gaps = [data[model]['accuracy_gap'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, model in enumerate(models):
        ax1.scatter(receptive_fields[i], test_accuracies[i], s=200, color=colors[i], 
                   label=model, alpha=0.7, edgecolors='black', linewidth=2)
    
    ax1.set_title('Влияние размера рецептивного поля на точность', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Размер рецептивного поля', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.68, 0.79)
    
    for i, model in enumerate(models):
        ax1.annotate(f'{test_accuracies[i]:.3f}', 
                    (receptive_fields[i], test_accuracies[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontweight='bold', fontsize=10)
    
    for i, model in enumerate(models):
        ax2.scatter(receptive_fields[i], accuracy_gaps[i], s=200, color=colors[i], 
                   label=model, alpha=0.7, edgecolors='black', linewidth=2)
    
    ax2.set_title('Влияние размера рецептивного поля на переобучение', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Размер рецептивного поля', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Gap (Train - Test)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.14, 0.17)
    
    for i, model in enumerate(models):
        ax2.annotate(f'{accuracy_gaps[i]:.3f}', 
                    (receptive_fields[i], accuracy_gaps[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_test_accuracy_and_training_time_bar():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.2.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = list(data.keys())
    test_accuracies = [data[model]['final_test_accuracy'] for model in models]
    training_times = [data[model]['training_time_seconds'] for model in models]
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, test_accuracies, width, label='Test Accuracy', 
                   color='lightblue', edgecolor='blue', linewidth=2, alpha=0.8)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, training_times, width, label='Время обучения (секунды)', 
                   color='lightcoral', edgecolor='red', linewidth=2, alpha=0.8)

    ax1.set_xlabel('Модели', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold', color='blue')
    ax2.set_ylabel('Время обучения (секунды)', fontsize=14, fontweight='bold', color='red')
    ax1.set_title('Сравнение Test Accuracy и времени обучения',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    
    ax1.set_ylim(0.65, 0.9)
    ax2.set_ylim(150, 300)
    
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', color='blue', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', color='red', fontsize=10)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def vanishing_exploding_gradients():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.2.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = list(data.keys())

    plt.figure(figsize=(14, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, model in enumerate(models):
        gradient_data = data[model]['gradient_norms_by_layer']
        
        gradient_norms = []
        layer_names = []
        
        for layer_name, norm in gradient_data.items():
            short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            layer_names.append(short_name)
            gradient_norms.append(norm)
        
        plt.plot(gradient_norms, marker='o', linewidth=3, markersize=8, 
                label=model, color=colors[i], alpha=0.8, markeredgecolor='black', markeredgewidth=0.5)

    plt.title('График норм градиентов по слоям', 
            fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Номер слоя', fontsize=12, fontweight='bold')
    plt.ylabel('Норма градиента', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Vanishing Gradients Threshold')
    plt.axhline(y=1e-1, color='orange', linestyle='--', alpha=0.7, label='Exploding Gradients Threshold')

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residual_effectiveness():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.2.JSON')
    
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    expected_models = ['ShallowCNN', 'MediumCNN', 'DeepCNN', 'ResidualCNN']
    for model in expected_models:
        if model not in data:
            raise ValueError(f"Модель '{model}' отсутствует в файле #2.2.JSON")
    
    colors = {
        'ShallowCNN': 'tab:blue',
        'MediumCNN': 'tab:orange',
        'DeepCNN': 'tab:green',
        'ResidualCNN': 'tab:red'
    }
    markers = {
        'ShallowCNN': 'o',
        'MediumCNN': 's',
        'DeepCNN': '^',
        'ResidualCNN': 'D'
    }

    sample_hist = data['ShallowCNN']['training_history']
    epochs = range(1, len(sample_hist['train_accuracies']) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for model_name in expected_models:
        hist = data[model_name]['training_history']
        plt.plot(
            epochs,
            hist['train_accuracies'],
            label=model_name,
            color=colors[model_name],
            marker=markers[model_name],
            markersize=4,
            linewidth=2
        )
    plt.title('Training Accuracy по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    for model_name in expected_models:
        hist = data[model_name]['training_history']
        plt.plot(
            epochs,
            hist['test_accuracies'],
            label=model_name,
            color=colors[model_name],
            marker=markers[model_name],
            markersize=4,
            linewidth=2
        )
    plt.title('Test Accuracy по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.suptitle('Сравнение моделей разной глубины и с Residual связями', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


def visualize_feature_maps_from_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем одно изображение из CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    image, label = test_dataset[0]  # Берём первое изображение
    image = image.unsqueeze(0).to(device)  # Добавляем batch dim

    # Определяем модели
    model_classes = {
        'ShallowCNN': ShallowCNN,
        'MediumCNN': MediumCNN,
        'DeepCNN': DeepCNN,
        'ResidualCNN': ResidualCNN
    }

    activations = {}

    for name, ModelClass in model_classes.items():
        model = ModelClass().to(device)
        model.eval()

        first_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                first_conv = module
                break

        activation = {}

        def hook_fn(m, i, o):
            activation['act'] = o.detach().cpu()

        handle = first_conv.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(image)
        handle.remove()

        activations[name] = activation['act'].squeeze(0).numpy()  # [C, H, W]

    max_channels = 16
    model_names = list(model_classes.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(n_models, max_channels + 1, 
                             figsize=(20, 3.5 * n_models),
                             gridspec_kw={'width_ratios': [1] + [1]*max_channels})

    img_for_plot = image.cpu().squeeze().permute(1, 2, 0).numpy()
    img_for_plot = img_for_plot * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
    img_for_plot = np.clip(img_for_plot, 0, 1)

    for row, model_name in enumerate(model_names):
        if max_channels + 1 > 0:
            axes[row, 0].imshow(img_for_plot)
            axes[row, 0].set_title("Input", fontsize=9, fontweight='bold') if row == 0 else None
            axes[row, 0].axis('off')

        axes[row, 0].text(-0.3, 0.5, model_name, 
                          transform=axes[row, 0].transAxes, 
                          fontsize=12, 
                          fontweight='bold',
                          rotation=90, 
                          va='center', 
                          ha='center')

        act = activations[model_name]
        n_ch = min(act.shape[0], max_channels)

        for c in range(n_ch):
            ax = axes[row, c + 1]
            ax.imshow(act[c], cmap='viridis', interpolation='nearest')
            ax.axis('off')
            if row == 0:
                ax.set_title(f'Ch {c+1}', fontsize=9)

        for c in range(n_ch, max_channels):
            axes[row, c + 1].axis('off')

    plt.suptitle('Feature Maps первого свёрточного слоя (одно и то же входное изображение)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.show()


# plot_accuracy_vs_training_time()
# plot_receptive_field_analysis()
# plot_test_accuracy_and_training_time_bar()
# vanishing_exploding_gradients()
# plot_residual_effectiveness()
# visualize_feature_maps_from_models()
