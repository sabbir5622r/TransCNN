import numpy as np

from evaluation.metrics import calculate_metrics
from utils.plotting import plot_evaluation


def evaluate_model(model, test_ds, model_name="Model", output_dir=None,
                   num_samples=12):
    y_true = []
    y_pred = []
    y_probs = []
    images_list = []

    test_ds.reset()
    steps = int(np.ceil(test_ds.samples / test_ds.batch_size))

    for _ in range(steps):
        images, labels = next(test_ds)
        probs = model.predict(images, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)

        y_probs.extend(probs.tolist())
        y_pred.extend(preds.tolist())
        y_true.extend(labels.flatten().astype(int).tolist())
        images_list.extend(images)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    metrics = calculate_metrics(y_true, y_pred)
    print(f"\n{model_name} Evaluation Results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    if output_dir is not None:
        plot_evaluation(y_true, y_pred, y_probs, images_list, model_name,
                        output_dir, num_samples)

    return metrics
