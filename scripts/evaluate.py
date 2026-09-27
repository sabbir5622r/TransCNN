import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing model weights")
    parser.add_argument("--model", required=True, choices=[
        "cnn", "cnn_cbam", "cnn_transformer", "cnn_cbam_transformer"])
    parser.add_argument("--dataset", required=True,
                        choices=["deepfake_dataset_1", "deepfake_dataset_2"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["test", "real_world"], default="test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--explain", choices=["gradcam", "gradcampp"])
    args = parser.parse_args()

    from data.dataset import create_evaluation_generator
    from models.build import build_model
    from evaluation.evaluate import evaluate_model
    from utils.config import load_config

    config = load_config(args.config)
    generator = create_evaluation_generator(
        Path(config["datasets"][args.dataset]) / args.split,
        config["image_size"], config["models"][args.model]["batch_size"],
    )
    model = build_model(args.model, config["image_size"])
    model.load_weights(args.weights)
    output = Path(__file__).resolve().parents[1] / "outputs" / args.dataset / args.model / args.split
    evaluate_model(model, generator, f"{args.model} ({args.dataset}, {args.split})",
                   output_dir=output if args.plots else None)
    if args.explain:
        from evaluation.explain import (
            get_gradcam_for_batch, get_gradcampp_for_batch, explanation_layer,
        )
        explain = get_gradcam_for_batch if args.explain == "gradcam" else get_gradcampp_for_batch
        images, labels = next(generator)
        explain(model, images, labels,
                layer_name=explanation_layer(model, args.model), num_samples=3,
                output_path=output / f"{args.explain}.png")


if __name__ == "__main__":
    main()
