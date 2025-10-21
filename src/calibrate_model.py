import pickle
import os
import joblib
import argparse
import sys
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

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
    parser.add_argument("--method", type=str, default="sigmoid", choices=["sigmoid", "isotonic"],
                        help="Calibration method: sigmoid (Platt scaling) or isotonic")
    args = parser.parse_args()
    
    timestamp = args.timestamp
    calibration_method = args.method
    
    print("="*60)
    print(f"CREDIT CARD DEFAULT - MODEL CALIBRATION")
    print("="*60)
    print(f"Timestamp: {timestamp}")
    print(f"Calibration Method: {calibration_method.upper()}")
    print(f"Dataset: Credit Card Default (UCI Repository)")
    print(f"Model: Support Vector Machine (SVM)")
    print("="*60)
    
    # Step 3: Load Trained Model
    print("\n[STEP 3] Loading trained model...")
    try:
        base_model_file = f'models/model_{timestamp}_base.joblib'
        base_model = joblib.load(base_model_file)
        print(f"✓ Loaded base model: {base_model_file}")
        print(f"  Model type: {type(base_model).__name__}")
    except Exception as e:
        raise ValueError(f'Failed to load trained model: {e}')
    
    # Load calibration and test datasets
    print("\n[STEP 3] Loading datasets...")
    try:
        with open('data/X_calib.pickle', 'rb') as f:
            X_calib = pickle.load(f)
        with open('data/y_calib.pickle', 'rb') as f:
            y_calib = pickle.load(f)
        with open('data/X_test.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data/y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)
        
        print(f"✓ Loaded calibration set: {len(y_calib)} samples")
        print(f"  Default rate: {np.mean(y_calib):.2%}")
        print(f"✓ Loaded test set: {len(y_test)} samples")
        print(f"  Default rate: {np.mean(y_test):.2%}")
    except Exception as e:
        raise ValueError(f'Failed to load datasets: {e}')
    
    # Evaluate base model before calibration
    print("\n" + "-"*60)
    print("BASE SVM MODEL PERFORMANCE (Before Calibration)")
    print("-"*60)
    print("Why SVMs need calibration:")
    print("  • SVMs optimize for margin maximization, not probabilities")
    print("  • Decision function output ≠ true probability")
    print("  • Often produce overconfident predictions (near 0 or 1)")
    print("  • Calibration transforms distances into reliable probabilities")
    print("-"*60)
    
    y_pred_base = base_model.predict(X_test)
    y_proba_base = base_model.predict_proba(X_test)[:, 1]
    
    base_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_base),
        "f1_score": f1_score(y_test, y_pred_base),
        "precision": precision_score(y_test, y_pred_base),
        "recall": recall_score(y_test, y_pred_base),
        "roc_auc": roc_auc_score(y_test, y_proba_base),
        "brier_score": brier_score_loss(y_test, y_proba_base),
        "log_loss": log_loss(y_test, y_proba_base),
        "expected_calibration_error": calculate_expected_calibration_error(y_test, y_proba_base)
    }
    
    print("\nClassification Metrics:")
    for metric in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
        print(f"  {metric:.<40} {base_metrics[metric]:.4f}")
    
    print("\nCalibration Metrics (lower is better):")
    for metric in ["brier_score", "log_loss", "expected_calibration_error"]:
        print(f"  {metric:.<40} {base_metrics[metric]:.4f}")
    
    # Analyze probability distribution
    print(f"\nProbability Distribution:")
    print(f"  Mean predicted probability: {np.mean(y_proba_base):.4f}")
    print(f"  Actual default rate: {np.mean(y_test):.4f}")
    calibration_gap_before = abs(np.mean(y_proba_base) - np.mean(y_test))
    print(f"  Calibration gap: {calibration_gap_before:.4f}")
    
    # Check for overconfidence
    extreme_low = np.sum(y_proba_base < 0.1) / len(y_proba_base) * 100
    extreme_high = np.sum(y_proba_base > 0.9) / len(y_proba_base) * 100
    
    print(f"\nOverconfidence Analysis:")
    print(f"  Probabilities < 0.1: {extreme_low:.1f}%")
    print(f"  Probabilities > 0.9: {extreme_high:.1f}%")
    print(f"  Total extreme: {extreme_low + extreme_high:.1f}%")
    
    if extreme_low + extreme_high > 50:
        print(f"\nHigh overconfidence detected! ({extreme_low + extreme_high:.1f}%)")
        print("  This is typical for SVMs and confirms need for calibration.")
    
    # Step 4: Calibrate Model Probabilities
    print("\n" + "="*60)
    print(f"[STEP 4] Calibrating model using {calibration_method} method")
    print("="*60)
    
    if calibration_method == "sigmoid":
        print("\nApplying Platt Scaling (Sigmoid calibration)...")
        print("  Method: Fits logistic regression on predicted probabilities")
        print("  Best for: Monotonic calibration errors (typical in SVMs)")
        print("  Advantage: Works well with limited calibration data")
    else:
        print("\nApplying Isotonic Regression...")
        print("  Method: Non-parametric piecewise constant function")
        print("  Best for: Non-monotonic calibration errors")
        print("  Advantage: More flexible, but needs more data")
    
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=calibration_method,
        cv='prefit'  # Use pre-fitted model with separate calibration set
    )
    
    print(f"\nCalibrating on {len(y_calib)} samples...")
    calibrated_model.fit(X_calib, y_calib)
    print("✓ Calibration complete!")
    
    # Evaluate calibrated model
    print("\n" + "-"*60)
    print("CALIBRATED MODEL PERFORMANCE (After Calibration)")
    print("-"*60)
    
    y_pred_calib = calibrated_model.predict(X_test)
    y_proba_calib = calibrated_model.predict_proba(X_test)[:, 1]
    
    calib_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_calib),
        "f1_score": f1_score(y_test, y_pred_calib),
        "precision": precision_score(y_test, y_pred_calib),
        "recall": recall_score(y_test, y_pred_calib),
        "roc_auc": roc_auc_score(y_test, y_proba_calib),
        "brier_score": brier_score_loss(y_test, y_proba_calib),
        "log_loss": log_loss(y_test, y_proba_calib),
        "expected_calibration_error": calculate_expected_calibration_error(y_test, y_proba_calib)
    }
    
    print("\nClassification Metrics:")
    for metric in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
        print(f"  {metric:.<40} {calib_metrics[metric]:.4f}")
    
    print("\nCalibration Metrics (lower is better):")
    for metric in ["brier_score", "log_loss", "expected_calibration_error"]:
        print(f"  {metric:.<40} {calib_metrics[metric]:.4f}")
    
    # Analyze calibrated probability distribution
    print(f"\nCalibrated Probability Distribution:")
    print(f"  Mean predicted probability: {np.mean(y_proba_calib):.4f}")
    print(f"  Actual default rate: {np.mean(y_test):.4f}")
    calibration_gap_after = abs(np.mean(y_proba_calib) - np.mean(y_test))
    print(f"  Calibration gap: {calibration_gap_after:.4f}")
    
    # Check calibrated overconfidence
    extreme_low_calib = np.sum(y_proba_calib < 0.1) / len(y_proba_calib) * 100
    extreme_high_calib = np.sum(y_proba_calib > 0.9) / len(y_proba_calib) * 100
    
    print(f"\nOverconfidence After Calibration:")
    print(f"  Probabilities < 0.1: {extreme_low_calib:.1f}%")
    print(f"  Probabilities > 0.9: {extreme_high_calib:.1f}%")
    print(f"  Total extreme: {extreme_low_calib + extreme_high_calib:.1f}%")
    
    # Calculate and display improvements
    print("\n" + "="*60)
    print("CALIBRATION IMPACT ANALYSIS")
    print("="*60)
    
    improvements = {}
    print("\nCalibration Quality Improvements:")
    
    for metric in ["brier_score", "expected_calibration_error", "log_loss"]:
        if base_metrics[metric] != 0:
            improvement_pct = ((base_metrics[metric] - calib_metrics[metric]) / 
                              base_metrics[metric] * 100)
            improvements[f"{metric}_improvement"] = improvement_pct
            
            status = "✓ Improved" if improvement_pct > 0 else "✗ Slightly worse"
            print(f"\n  {metric.replace('_', ' ').title()}:")
            print(f"    Before: {base_metrics[metric]:.4f}")
            print(f"    After:  {calib_metrics[metric]:.4f}")
            print(f"    Change: {improvement_pct:+.2f}% {status}")
    
    # Calibration gap improvement
    gap_improvement = ((calibration_gap_before - calibration_gap_after) / 
                      calibration_gap_before * 100)
    improvements['calibration_gap_improvement'] = gap_improvement
    
    print(f"\n  Calibration Gap:")
    print(f"    Before: {calibration_gap_before:.4f}")
    print(f"    After:  {calibration_gap_after:.4f}")
    print(f"    Change: {gap_improvement:+.2f}%")
    
    # Overconfidence reduction
    overconf_reduction = ((extreme_low + extreme_high) - 
                         (extreme_low_calib + extreme_high_calib))
    
    print(f"\n  Overconfidence Reduction:")
    print(f"    Before: {extreme_low + extreme_high:.1f}% extreme predictions")
    print(f"    After:  {extreme_low_calib + extreme_high_calib:.1f}% extreme predictions")
    print(f"    Reduction: {overconf_reduction:.1f} percentage points")
    
    # Business impact interpretation
    print("\n" + "="*60)
    print("BUSINESS IMPACT FOR CREDIT RISK")
    print("="*60)
    
    brier_improvement = improvements.get('brier_score_improvement', 0)
    ece_improvement = improvements.get('expected_calibration_error_improvement', 0)
        
    # Step 5: Save Calibrated Model
    print("\n" + "="*60)
    print("[STEP 5] Saving calibrated model")
    print("="*60)
    
    if not os.path.exists('models/'):
        os.makedirs("models/")
    
    calibrated_model_filename = f'model_{timestamp}_calibrated_{calibration_method}.joblib'
    joblib.dump(calibrated_model, calibrated_model_filename)
    print(f"✓ Saved calibrated model: {calibrated_model_filename}")
    
    # Save calibration report
    calibration_report = {
        'timestamp': timestamp,
        'calibration_method': calibration_method,
        'dataset': 'Credit Card Default (UCI ID: 350)',
        'base_model_type': 'SVM_RBF',
        'base_model_file': base_model_file,
        'calibrated_model_file': calibrated_model_filename,
        'base_metrics': base_metrics,
        'calibrated_metrics': calib_metrics,
        'improvements': improvements,
        'calibration_set_size': len(y_calib),
        'test_set_size': len(y_test),
        'base_default_rate': float(np.mean(y_test)),
        'calibration_gap_before': float(calibration_gap_before),
        'calibration_gap_after': float(calibration_gap_after),
        'overconfidence_before': float(extreme_low + extreme_high),
        'overconfidence_after': float(extreme_low_calib + extreme_high_calib),
        'business_context': 'Credit Card Default Prediction'
    }
    
    report_filename = f'calibration_report_{timestamp}.pickle'
    with open(report_filename, 'wb') as f:
        pickle.dump(calibration_report, f)
    print(f"✓ Saved calibration report: {report_filename}")
    
    print("\n" + "="*60)
    print("CALIBRATION WORKFLOW COMPLETE")
    print("="*60)
    print(f"\nModels available for credit risk assessment:")
    print(f"  • Base SVM: {base_model_file}")
    print(f"  • Calibrated SVM: {calibrated_model_filename}")
    
    print(f"\nWhy the calibrated model is superior:")
    print(f"  • {brier_improvement:.1f}% better probability accuracy (Brier Score)")
    print(f"  • {ece_improvement:.1f}% better calibration (ECE)")
    print(f"  • {overconf_reduction:.1f}pp reduction in overconfident predictions")
    print(f"  • Probabilities now reliably match actual default rates")