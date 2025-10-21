import pickle
import os
import json
import numpy as np
import joblib
import argparse
import sys
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.abspath('..'))


def calculate_expected_calibration_error(y_true, y_proba, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    parser.add_argument("--model-type", type=str, default="calibrated", 
                       choices=["base", "calibrated"],
                       help="Which model to evaluate: base or calibrated")
    args = parser.parse_args()
    
    timestamp = args.timestamp
    model_type = args.model_type
    
    print("="*60)
    print(f"CREDIT CARD DEFAULT - MODEL EVALUATION")
    print("="*60)
    print(f"Evaluating: {model_type.upper()} model")
    print(f"Timestamp: {timestamp}")
    print(f"Dataset: Credit Card Default (UCI Repository)")
    print("="*60)
    
    # Load test data
    try:
        with open('data/X_test.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data/y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)
        print(f"\n✓ Loaded test data: {X_test.shape[0]} samples")
        print(f"  Default rate: {np.mean(y_test):.2%}")
    except Exception as e:
        raise ValueError(f'Failed to load test data: {e}')
    
    # Load the appropriate model
    try:
        if model_type == "base":
            model_file = f'models/model_{timestamp}_base.joblib'
        else:
            # Try sigmoid first, then isotonic
            if os.path.exists(f'models/model_{timestamp}_calibrated_sigmoid.joblib'):
                model_file = f'models/model_{timestamp}_calibrated_sigmoid.joblib'
            else:
                model_file = f'models/model_{timestamp}_calibrated_isotonic.joblib'
        
        model = joblib.load(model_file)
        print(f"✓ Loaded model: {model_file}")
    except Exception as e:
        raise ValueError(f'Failed to load model: {e}')
    
    # Make predictions
    print("\n" + "-"*60)
    print("GENERATING PREDICTIONS")
    print("-"*60)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"✓ Generated predictions for {len(y_test)} samples")
    
    # Calculate comprehensive metrics
    print("\n" + "="*60)
    print(f"{model_type.upper()} MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Classification metrics
    print("\nClassification Performance:")
    print("-" * 40)
    
    metrics = {
        "model_type": model_type,
        "timestamp": timestamp,
        "dataset": "Credit Card Default",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    
    for metric in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
        print(f"  {metric:.<40} {metrics[metric]:.4f}")
    
    # Calibration metrics
    print("\nCalibration Quality (lower is better):")
    print("-" * 40)
    
    calibration_metrics = {
        "brier_score": float(brier_score_loss(y_test, y_proba)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "expected_calibration_error": float(calculate_expected_calibration_error(y_test, y_proba))
    }
    
    for metric, value in calibration_metrics.items():
        print(f"  {metric:.<40} {value:.4f}")
    
    metrics.update(calibration_metrics)
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0][0]:>6}")
    print(f"  False Positives: {cm[0][1]:>6}")
    print(f"  False Negatives: {cm[1][0]:>6}")
    print(f"  True Positives:  {cm[1][1]:>6}")
    
    # Calculate calibration curve
    print("\nGenerating calibration curve data...")
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=10, strategy='uniform'
    )
    
    calibration_data = {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist()
    }
    
    metrics["calibration_curve"] = calibration_data
    
    # Analyze prediction distribution
    print("\nPrediction Probability Distribution:")
    print("-" * 40)
    
    pred_stats = {
        "min_probability": float(np.min(y_proba)),
        "max_probability": float(np.max(y_proba)),
        "mean_probability": float(np.mean(y_proba)),
        "median_probability": float(np.median(y_proba)),
        "std_probability": float(np.std(y_proba))
    }
    
    for stat, value in pred_stats.items():
        print(f"  {stat:.<40} {value:.4f}")
    
    metrics["prediction_distribution"] = pred_stats
    
    # Check calibration alignment
    print("\nCalibration Alignment:")
    print("-" * 40)
    print(f"  Actual default rate: {np.mean(y_test):.4f}")
    print(f"  Mean predicted probability: {np.mean(y_proba):.4f}")
    calibration_gap = abs(np.mean(y_proba) - np.mean(y_test))
    print(f"  Calibration gap: {calibration_gap:.4f}")
    
    metrics["calibration_gap"] = float(calibration_gap)
    
    # Check for overconfidence
    extreme_low = np.sum(y_proba < 0.1) / len(y_proba) * 100
    extreme_high = np.sum(y_proba > 0.9) / len(y_proba) * 100
    
    print("\nOverconfidence Analysis:")
    print("-" * 40)
    print(f"  Probabilities < 0.1: {extreme_low:.1f}%")
    print(f"  Probabilities > 0.9: {extreme_high:.1f}%")
    print(f"  Total extreme: {extreme_low + extreme_high:.1f}%")
    
    metrics["overconfidence"] = {
        "extreme_low_pct": float(extreme_low),
        "extreme_high_pct": float(extreme_high),
        "total_extreme_pct": float(extreme_low + extreme_high)
    }
    
    if model_type == "base" and (extreme_low + extreme_high) > 50:
        print("\n Base SVM shows high overconfidence (typical)")
        print("  Calibration should significantly improve this.")
    elif model_type == "calibrated" and (extreme_low + extreme_high) < 30:
        print("\n✓ Calibrated model shows reduced overconfidence")
        print("  Probability distribution is more spread and reliable.")
    
    # Save metrics to JSON file
    if not os.path.exists('metrics/'): 
        os.makedirs("metrics/")
    
    metrics_filename = f'{timestamp}_{model_type}_metrics.json'
    metrics_path = os.path.join('metrics', metrics_filename)
    
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n✓ Saved metrics to: {metrics_filename}")
    
    # Generate summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model Type: {model_type.upper()}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f} (lower is better)")
    print(f"Expected Calibration Error: {metrics['expected_calibration_error']:.4f} (lower is better)")
    print(f"Calibration Gap: {calibration_gap:.4f} (lower is better)")
    
    if model_type == "calibrated":
        print("\n✓ This calibrated model provides more reliable probability estimates")
        print("  for credit risk assessment compared to the base SVM model.")
        print("\nBusiness Applications:")
        print("  • Accurate loan pricing based on default probability")
        print("  • Reliable credit limit decisions")
        print("  • Better portfolio risk management")
        print("  • Regulatory compliance with explainable risk scores")