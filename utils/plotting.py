from pathlib import Path
import random

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_evaluation(y_true, y_pred, y_probs, images_list, model_name,
                    output_dir, num_samples=12):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    maroon_cmap = LinearSegmentedColormap.from_list("maroon_cmap", ["white", "purple"])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=maroon_cmap, xticklabels=["DeepFake", "Real"], yticklabels=["DeepFake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(output_dir / "confusion.png", bbox_inches="tight", dpi=300)    
    plt.show()

    # ROC curves for both classes and mean
    y_probs_class0 = 1 - y_probs  # for class 0
    fpr0, tpr0, _ = roc_curve(y_true == 0, y_probs_class0)
    roc_auc0 = auc(fpr0, tpr0)

    fpr1, tpr1, _ = roc_curve(y_true == 1, y_probs)
    roc_auc1 = auc(fpr1, tpr1)

    fpr_mean, tpr_mean, _ = roc_curve(y_true, y_probs)
    roc_auc_mean = auc(fpr_mean, tpr_mean)

    plt.figure(figsize=(8,6))
    plt.plot(fpr0, tpr0, color='red', lw=2, label=f'ROC Curve for Fake (AUC = {roc_auc0:.2f})')
    plt.plot(fpr1, tpr1, color='green', lw=2, label=f'ROC Curve for Real (AUC = {roc_auc1:.2f})')
    plt.plot(fpr_mean, tpr_mean, color='blue', lw=2, linestyle='--', label=f'Mean ROC (AUC = {roc_auc_mean:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "roc.png", bbox_inches="tight", dpi=300)    
    plt.show()

    # Random sample predictions visualization
    num_samples = min(num_samples, len(images_list))
    indices = random.sample(range(len(images_list)), num_samples)

    fig, axes = plt.subplots(2, 6, figsize=(17, 7))  # 2 rows × 6 columns
    axes = axes.flatten()
    class_labels = ["DeepFake", "Real"]

    for i, idx in enumerate(indices):
        ax = axes[i]
        img = images_list[idx]
        if img.max() > 1.0:
            img = img / 255.0
        ax.imshow(img)

        true_label = class_labels[int(y_true[idx])]
        pred_label = class_labels[int(y_pred[idx])]
        confidence = y_probs[idx]

        ax.set_title(f"Pred: {pred_label}\nActual: {true_label}\nConfidence: {confidence:.2f}", 
                     color="black", fontsize=10)
        ax.axis("off")

    # Turn off unused axes
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    #plt.subplots_adjust(wspace=-0.3, hspace=0.2)
    plt.savefig(output_dir / "predictions.png", bbox_inches="tight", dpi=300)
    plt.show()


def plot_history(history, output_path, figsize=(10, 3)):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    #Extract training history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot Training & Validation Accuracy
    ax1.plot(epochs, acc, 'b', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'orange', label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot Training & Validation Loss
    ax2.plot(epochs, loss, 'b', label='Training Loss')
    ax2.plot(epochs, val_loss, 'orange', label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
