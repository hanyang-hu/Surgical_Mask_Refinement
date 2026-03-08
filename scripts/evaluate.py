"""Evaluation script for computing metrics on refined masks.

Computes segmentation metrics (IoU, Dice, precision, recall, etc.)
comparing predicted refined masks against ground truth.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate refined masks")
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory containing predicted masks"
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        required=True,
        help="Directory containing ground truth refined masks"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for metrics"
    )
    # TODO: Add more arguments
    return parser.parse_args()


def compute_metrics(pred_masks, gt_masks):
    """Compute segmentation metrics.
    
    TODO: Implement IoU, Dice, precision, recall
    TODO: Support per-sample and aggregate metrics
    """
    pass


def main():
    """Main evaluation function.
    
    TODO: Load predicted and ground truth masks
    TODO: Compute metrics
    TODO: Print and save results
    """
    args = parse_args()
    
    print("=" * 50)
    print("Evaluation")
    print("=" * 50)
    print(f"Predictions: {args.predictions_dir}")
    print(f"Ground truth: {args.ground_truth_dir}")
    
    # TODO: Implement evaluation
    

if __name__ == "__main__":
    main()
