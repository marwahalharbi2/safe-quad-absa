#!/usr/bin/env python3
"""
SAFE Dataset Evaluation Script
==============================

Command-line tool for evaluating Quad-ABSA datasets using the SAFE metric.

Usage:
    python evaluate_dataset.py input.csv output.csv --tau 70 --model all-MiniLM-L6-v2

Example:
    python evaluate_dataset.py \
        data/human_eval.csv \
        results/safe_output.csv \
        --tau 70.0 \
        --review-id-col Review_ID \
        --verbose
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from safe_metric import SAFEMetric, QuadEvaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Quad-ABSA predictions using SAFE metric",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python evaluate_dataset.py input.csv output.csv
  
  # Custom threshold and model
  python evaluate_dataset.py input.csv output.csv --tau 75 --model all-mpnet-base-v2
  
  # Custom column names
  python evaluate_dataset.py input.csv output.csv \\
      --gold-cols Aspect Category Opinion Sentiment \\
      --pred-cols Aspect_Pred Category_Pred Opinion_Pred Sentiment_Pred
        """
    )
    
    # Required arguments
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input CSV file with gold and predicted quads"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save output CSV with SAFE scores and classifications"
    )
    
    # Optional arguments
    parser.add_argument(
        "--tau",
        type=float,
        default=70.0,
        help="Similarity threshold for TP classification (0-100). Default: 70.0"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name. Default: all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--review-id-col",
        type=str,
        default="Review_ID",
        help="Name of review ID column. Default: Review_ID"
    )
    parser.add_argument(
        "--gold-cols",
        nargs=4,
        default=["Aspect", "Category", "Opinion", "Sentiment"],
        metavar=("ASPECT", "CATEGORY", "OPINION", "SENTIMENT"),
        help="Gold standard column names. Default: Aspect Category Opinion Sentiment"
    )
    parser.add_argument(
        "--pred-cols",
        nargs=4,
        default=["Aspect.1", "Category.1", "Opinion.1", "Sentiment.1"],
        metavar=("ASPECT", "CATEGORY", "OPINION", "SENTIMENT"),
        help="Prediction column names. Default: Aspect.1 Category.1 Opinion.1 Sentiment.1"
    )
    parser.add_argument(
        "--append-metrics",
        action="store_true",
        help="Append global metrics summary to output CSV"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    return parser.parse_args()


def print_banner():
    """Print startup banner."""
    print("=" * 80)
    print("SAFE: Semantic-Aware Flexible Evaluation for Quad-ABSA")
    print("=" * 80)


def print_metrics(metrics: dict, tau: float):
    """Print evaluation metrics in formatted table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Threshold (τ):     {tau:.2f}")
    print(f"  Total Reviews:     {metrics['num_reviews']}")
    print()
    print("  Classification Counts:")
    print(f"    TP: {metrics['total_tp']:4d}  (Model matched gold, SAFE ≥ {tau:.1f})")
    print(f"    FP: {metrics['total_fp']:4d}  (Model wrong or no gold match)")
    print(f"    FN: {metrics['total_fn']:4d}  (Gold exists, no model match)")
    print()
    print("  Global Metrics:")
    print(f"    Precision: {metrics['precision']:6.2f}%")
    print(f"    Recall:    {metrics['recall']:6.2f}%")
    print(f"    F1:        {metrics['f1']:6.2f}%")
    print("=" * 80)


def main():
    """Main execution function."""
    args = parse_args()
    
    if args.verbose:
        print_banner()
    
    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    if args.verbose:
        print(f"\n[1/4] Loading data from: {args.input_file}")
    
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"      Loaded {len(df)} rows")
        unique_reviews = df[args.review_id_col].dropna().nunique()
        print(f"      Found {unique_reviews} unique reviews")
    
    # Initialize SAFE metric
    if args.verbose:
        print(f"\n[2/4] Initializing SAFE metric")
        print(f"      Model: {args.model}")
        print(f"      Threshold (τ): {args.tau}")
    
    try:
        safe_metric = SAFEMetric(
            model_name=args.model,
            tau=args.tau
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = QuadEvaluator(
        safe_metric=safe_metric,
        gold_columns=args.gold_cols,
        pred_columns=args.pred_cols
    )
    
    # Run evaluation
    if args.verbose:
        print(f"\n[3/4] Evaluating dataset...")
    
    try:
        results_df, metrics = evaluator.evaluate_dataset(
            df,
            review_id_column=args.review_id_col
        )
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print results
    if args.verbose:
        print_metrics(metrics, args.tau)
    
    # Save output
    if args.verbose:
        print(f"\n[4/4] Saving results to: {args.output_file}")
    
    try:
        # Create output directory if needed
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_df.to_csv(output_path, index=False)
        
        # Optionally append metrics summary
        if args.append_metrics:
            with open(output_path, 'a') as f:
                f.write('\n\nGlobal Metrics Summary\n')
                f.write('Metric,Value\n')
                f.write(f'Total_TP,{metrics["total_tp"]}\n')
                f.write(f'Total_FP,{metrics["total_fp"]}\n')
                f.write(f'Total_FN,{metrics["total_fn"]}\n')
                f.write(f'Global_Precision,{metrics["precision"]:.2f}\n')
                f.write(f'Global_Recall,{metrics["recall"]:.2f}\n')
                f.write(f'Global_F1,{metrics["f1"]:.2f}\n')
                f.write(f'Threshold,{args.tau}\n')
                f.write(f'Model,{args.model}\n')
                f.write(f'Timestamp,{datetime.now().isoformat()}\n')
        
        if args.verbose:
            print(f"      Successfully saved results")
            print(f"\n✓ Evaluation complete!")
    
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
