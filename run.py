import argparse
from od_model.pipelines import (
    training_pipeline,
    inference_pipeline,
    evaluation_pipeline,
    visualize_pipeline
)
from od_model.utils import get_logger, load_config

logger = get_logger()


def main(args):
    cfg = load_config("od_model/config1.yaml")

    if args.train:
        logger.info("[INFO] Starting training pipeline...")
        training_pipeline(cfg=cfg)

    if args.inference:
        logger.info("[INFO] Running inference...")
        if not args.model_path:
            logger.error("Model path required for inference.")
            raise ValueError("Model path required for inference.")
        inference_pipeline(cfg=cfg, model_path=args.model_path)
    
    if args.visualize:
        logger.info("[INFO] Running visualization pipeline...")
        visualize_pipeline(
            cfg=cfg,
            json_path="artifacts/predictions.json",
            num_samples=10
            )

    if args.eval:
        logger.info("[INFO] Running evaluation...")
        evaluation_pipeline(
            cfg=cfg,
            json_path="artifacts/predictions.json",
            num_classes=10,
            logger=logger
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDD100K Object Detection Pipeline")

    parser.add_argument(
        "--train", action="store_true", help="Run training pipeline"
    )
    parser.add_argument(
        "--inference", action="store_true", help="Run inference pipeline"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Run visualization pipeline"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation pipeline"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to trained model weights (.pth)"
    )

    args = parser.parse_args()
    main(args)
