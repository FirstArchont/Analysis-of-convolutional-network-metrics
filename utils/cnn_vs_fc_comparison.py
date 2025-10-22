import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_final_accuracies_comparison():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_MNIST.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = list(data.keys())
    train_accuracies = [data[model]['final_train_accuracy'] for model in models]
    test_accuracies = [data[model]['final_test_accuracy'] for model in models]
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_accuracies, width, label='Train Accuracy', 
                   color='skyblue', edgecolor='navy', linewidth=1.2, alpha=0.8)
    bars2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', 
                   color='lightcoral', edgecolor='darkred', linewidth=1.2, alpha=0.8)
    
    ax.set_xlabel('Модели', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Итоговые Train и Test Accuracy сравнение на MNIST',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0.95, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    ax.legend(fontsize=12, loc='lower right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_training_inference_comparison():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_MNIST.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = list(data.keys())
    training_times = [data[model]['training_time_seconds'] for model in models]
    inference_times = [data[model]['inference_time_ms'] for model in models]
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, training_times, width, label='Время обучения (с)', 
                   color='lightgreen', edgecolor='green', linewidth=1.5)
    bars2 = plt.bar(x + width/2, inference_times, width, label='Время инференса (мс)', 
                   color='orange', edgecolor='darkorange', linewidth=1.5)
    
    plt.xlabel('Модели', fontsize=12, fontweight='bold')
    plt.ylabel('Время', fontsize=12, fontweight='bold')
    plt.title('Сравнение времени обучения и времени инференса',
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, models)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (max(training_times) * 0.01),
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_parameters_comparison():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_MNIST.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = list(data.keys())
    parameters = [data[model]['parameters'] for model in models]
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(models, parameters, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    plt.xlabel('Модели', fontsize=12, fontweight='bold')
    plt.ylabel('Количество параметров', fontsize=12, fontweight='bold')
    plt.title('Сравнение количества параметров моделей', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(parameters) * 0.01),
                f'{height:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_training_time_barchart():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_CIFAR.JSON')
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
    bars2 = ax2.bar(x + width/2, training_times, width, label='Время обучения (с)', 
                   color='lightcoral', edgecolor='red', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Модели', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold', color='blue')
    ax2.set_ylabel('ВРемя обучения (секунды)', fontsize=14, fontweight='bold', color='red')
    ax1.set_title('Сравнение Test Accuracy и времени обучения', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', color='blue')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.01,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', color='red')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_overfitting_analysis():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_CIFAR.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = list(data.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    accuracy_gaps = []
    loss_gaps = []
    
    for model in models:
        overfitting_data = data[model]['overfitting_analysis']
        accuracy_gaps.append(overfitting_data['final_accuracy_gap'])
        loss_gaps.append(overfitting_data['final_loss_gap'])
    
    bars1 = ax1.bar(models, accuracy_gaps, color='lightcoral', edgecolor='red', linewidth=2, alpha=0.8)
    ax1.set_title('Overfitting Analysis: Accuracy Gap\n(Train Accuracy - Test Accuracy)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Разница точностей', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Модели', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', color='red')
    
    bars2 = ax2.bar(models, loss_gaps, color='lightblue', edgecolor='blue', linewidth=2, alpha=0.8)
    ax2.set_title('Overfitting Analysis: Loss Gap\n(Test Loss - Train Loss)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Разница потерь', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Модели', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', color='blue')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_CIFAR.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    models = list(data.keys())
    
    fig, axes = plt.subplots(1, len(models), figsize=(20, 6))
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        cm_data = data[model]['confusion_matrix_data']
        confusion_mat = np.array(cm_data['confusion_matrix'])
        
        cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        
        im = axes[i].imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].set_title(f'{model}\nConfusion Matrix', fontsize=14, fontweight='bold', pad=20)
        
        plt.colorbar(im, ax=axes[i])
        
        tick_marks = np.arange(len(cifar10_classes))
        axes[i].set_xticks(tick_marks)
        axes[i].set_yticks(tick_marks)
        axes[i].set_xticklabels(cifar10_classes, rotation=45, ha='right')
        axes[i].set_yticklabels(cifar10_classes)
        axes[i].set_xlabel('Угаданные классы', fontsize=12)
        axes[i].set_ylabel('Истинные классы', fontsize=12)
        
        thresh = cm_normalized.max() / 2.
        for i_row in range(len(cifar10_classes)):
            for j_col in range(len(cifar10_classes)):
                axes[i].text(j_col, i_row, f'{confusion_mat[i_row, j_col]}\n({cm_normalized[i_row, j_col]:.2f})',
                           ha="center", va="center",
                           color="white" if cm_normalized[i_row, j_col] > thresh else "black",
                           fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_gradient_norms():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_CIFAR.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = list(data.keys())
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, model in enumerate(models):
        gradient_data = data[model]['gradient_flow']
        
        gradient_norms = []
        layer_names = []
        
        for layer_name, gradients in gradient_data.items():
            short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            layer_names.append(short_name)
            gradient_norms.append(gradients['norm'])
        
        plt.plot(gradient_norms, marker='o', linewidth=3, markersize=8, 
                label=model, color=colors[i], alpha=0.8, markeredgecolor='black', markeredgewidth=0.5)
    
    plt.title('График норм градиентов по слоям', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Номер слоя', fontsize=12, fontweight='bold')
    plt.ylabel('Норма градиента', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.axhline(y=1e-7, color='red', linestyle='--', alpha=0.7, label='Vanishing Gradients Threshold')
    plt.axhline(y=1000, color='orange', linestyle='--', alpha=0.7, label='Exploding Gradients Threshold')
    
    plt.legend()
    plt.tight_layout()
    plt.show()


# plot_final_accuracies_comparison()
# plot_training_inference_comparison()
# plot_parameters_comparison()
# plot_accuracy_vs_training_time_barchart()
# plot_overfitting_analysis()
# plot_confusion_matrices()
# plot_gradient_norms()
