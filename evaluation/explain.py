from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import tensorflow as tf


def get_gradcam_for_batch(model, images, labels, layer_name="gradcam_conv", class_labels=["Class 0", "Class 1"], num_samples=4, output_path=None):
    num_samples = min(num_samples, len(images))

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    images = tf.convert_to_tensor(images)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        img_tensor = tf.expand_dims(images[i], axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            pred_index = tf.argmax(predictions[0]) if predictions.shape[-1] > 1 else 0
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None or conv_outputs is None:
            print(f"[Warning] No gradients or conv outputs for image {i}. Skipping.")
            continue

        grads = grads[0]
        conv_outputs = conv_outputs[0]

        # Normalize gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        try:
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            if tf.math.equal(max_val, 0):
                raise ValueError("Max value of heatmap is zero.")
            heatmap /= max_val
            heatmap = heatmap.numpy()
        except Exception as e:
            print(f"[Error] Heatmap computation failed for image {i}: {e}")
            continue

        img = images[i].numpy()
        if img.max() <= 1:
            img_vis = np.uint8(img * 255)
        else:
            img_vis = np.uint8(img)

        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

        try:
            heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        except Exception as e:
            print(f"[Error] Failed to resize heatmap at index {i}: {e}")
            continue

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Plot
        axes[i][0].imshow(img_vis)
        axes[i][0].axis("off")
        axes[i][1].imshow(heatmap_resized, cmap='jet')
        axes[i][1].axis("off")
        axes[i][2].imshow(overlay_rgb)
        axes[i][2].axis("off")

        titles = ["Original Image", "Grad-CAM Heatmap", "Grad-CAM Overlay"]
        for j in range(3):
            ax = axes[i][j]
            ax.add_patch(patches.Rectangle((0, 1.02), 1, 0.12, transform=ax.transAxes, clip_on=False, color='red'))
            ax.text(0.5, 1.08, titles[j], transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='white', fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.2, hspace=0.15)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()

def get_gradcampp_for_batch(model, images, labels, layer_name="gradcam_conv", class_labels=["Class 0", "Class 1"], num_samples=4, output_path=None):
    num_samples = min(num_samples, len(images))

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    images = tf.convert_to_tensor(images)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        img_tensor = tf.expand_dims(images[i], axis=0)

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_outputs, predictions = grad_model(img_tensor)
                    pred_index = tf.argmax(predictions[0]) if predictions.shape[-1] > 1 else 0
                    loss = predictions[:, pred_index]

                grads = tape3.gradient(loss, conv_outputs)
            grads2 = tape2.gradient(grads, conv_outputs)
        grads3 = tape1.gradient(grads2, conv_outputs)

        if grads is None or conv_outputs is None:
            print(f"[Warning] No gradients or conv outputs for image {i}. Skipping.")
            continue

        conv_outputs = conv_outputs[0]
        grads = grads[0]
        grads2 = grads2[0]
        grads3 = grads3[0]

        # Grad-CAM++ alpha coefficients
        numerator = grads2
        denominator = 2.0 * grads2 + grads3 * conv_outputs
        denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))
        alphas = numerator / denominator
        alphas = tf.nn.relu(alphas)

        # Normalize alphas
        weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))

        try:
            heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.reduce_max(heatmap)
            if tf.math.equal(max_val, 0):
                raise ValueError("Max value of heatmap is zero.")
            heatmap /= max_val
            heatmap = heatmap.numpy()
        except Exception as e:
            print(f"[Error] Heatmap computation failed for image {i}: {e}")
            continue

        img = images[i].numpy()
        if img.max() <= 1:
            img_vis = np.uint8(img * 255)
        else:
            img_vis = np.uint8(img)

        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

        try:
            heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        except Exception as e:
            print(f"[Error] Failed to resize heatmap at index {i}: {e}")
            continue

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Plot
        axes[i][0].imshow(img_vis)
        axes[i][0].axis("off")
        axes[i][1].imshow(heatmap_resized, cmap='jet')
        axes[i][1].axis("off")
        axes[i][2].imshow(overlay_rgb)
        axes[i][2].axis("off")

        titles = ["Original Image", "Grad-CAM++ Heatmap", "Grad-CAM++ Overlay"]
        for j in range(3):
            ax = axes[i][j]
            ax.add_patch(patches.Rectangle((0, 1.02), 1, 0.12, transform=ax.transAxes, clip_on=False, color='red'))
            ax.text(0.5, 1.08, titles[j], transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='white', fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.2, hspace=0.15)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()


def explanation_layer(model, model_name):
    if model_name == "cnn_transformer":
        # The original conv2d_7 was the fourth backbone convolution.
        return [layer for layer in model.layers
                if isinstance(layer, tf.keras.layers.Conv2D)][3].name
    return "gradcam_conv"
