import argparse
from pathlib import Path


def train_main(model_name):
    parser = argparse.ArgumentParser(description=f"Train {model_name}")
    parser.add_argument("--dataset", required=True,
                        choices=["deepfake_dataset_1", "deepfake_dataset_2"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--real-world", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    args = parser.parse_args()

    from data.dataset import create_generators
    from models.build import build_model
    from training.train import train_model
    from evaluation.evaluate import evaluate_model
    from utils.config import load_config
    from utils.plotting import plot_history

    config = load_config(args.config)
    settings = config["models"][model_name]
    generators = create_generators(
        config["datasets"][args.dataset], config["image_size"],
        settings["batch_size"], config["augmentation"],
        include_real_world=args.real_world,
    )
    model = build_model(model_name, config["image_size"])
    model.summary()
    history = train_model(model, generators["train"], generators["val"],
                          config, model_name)
    output = Path(__file__).resolve().parents[1] / "outputs" / args.dataset / model_name
    if args.save_weights:
        output.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(output / "model.weights.h5"))
    if args.plots:
        plot_history(history, output / "learning_wide.png", figsize=(10, 3))
        plot_history(history, output / "learning.png", figsize=(8, 6))
    for split in ["test"] + (["real_world"] if args.real_world else []):
        evaluate_model(
            model, generators[split], f"{model_name} ({args.dataset}, {split})",
            output_dir=output / split if args.plots else None,
        )
