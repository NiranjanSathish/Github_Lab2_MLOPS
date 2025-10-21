import mlflow
import datetime
import os
import pickle
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
import argparse
import sys
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    print("\n" + "="*60)
    print("LOADING CREDIT CARD DEFAULT DATASET")
    print("="*60)
    
    # Fetch dataset from UCI repository
    try:
        print("Fetching dataset from UCI ML Repository...")
        default_of_credit_card_clients = fetch_ucirepo(id=350)
        
        # Extract features and target
        X = default_of_credit_card_clients.data.features
        y = default_of_credit_card_clients.data.targets
        
        # Convert to numpy arrays and handle target
        X = X.values
        y = y.values.ravel()  # Flatten target array
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Default rate: {np.mean(y):.2%}")
        
        # Print dataset information
        print("\nDataset Information:")
        print(f"  Name: {default_of_credit_card_clients.metadata.get('name', 'Credit Card Default')}")
        print(f"  Task: Binary Classification (Default Prediction)")
        print(f"  Domain: Financial Risk Assessment")
        
    except Exception as e:
        raise ValueError(f"Failed to load dataset from UCI repository: {e}")
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Remove any rows with missing values
    valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_indices]
    y = y[valid_indices]
    print(f"✓ Cleaned data: {X.shape[0]} samples retained")
    
    # Split data: train (60%), calibration (20%), test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData Split:")
    print(f"  Training set: {len(y_train)} samples ({np.mean(y_train):.2%} default)")
    print(f"  Calibration set: {len(y_calib)} samples ({np.mean(y_calib):.2%} default)")
    print(f"  Test set: {len(y_test)} samples ({np.mean(y_test):.2%} default)")
    
    # Feature scaling (CRITICAL for SVM)
    print("\nApplying standardization (critical for SVM)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features standardized (mean=0, std=1)")
    
    # Save all datasets and scaler for later use
    if not os.path.exists('data'): 
        os.makedirs('data/')
    
    datasets = {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_calib': X_calib_scaled,
        'y_calib': y_calib,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler
    }
    
    for name, data in datasets.items():
        with open(f'data/{name}.pickle', 'wb') as f:
            pickle.dump(data, f)
    
    print("✓ Datasets and scaler saved")
    
    # MLflow setup
    print("\n" + "="*60)
    print("MODEL TRAINING - SUPPORT VECTOR MACHINE")
    print("="*60)
    
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Credit Card Default"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"    
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{dataset_name}_SVM_training"):
        
        # Log parameters
        params = {
            "dataset_name": dataset_name,
            "dataset_id": 350,
            "total_samples": X.shape[0],
            "n_features": X.shape[1],
            "train_size": X_train_scaled.shape[0],
            "calibration_size": X_calib_scaled.shape[0],
            "test_size": X_test_scaled.shape[0],
            "default_rate": float(np.mean(y)),
            "model_type": "SVM_RBF",
            "scaling": "StandardScaler",
            "kernel": "rbf",
            "class_weight": "balanced"
        }
        mlflow.log_params(params)
        
        # Train SVM model
        print("Training Support Vector Machine (SVM)...")
        print("  Kernel: RBF (Radial Basis Function)")
        print("  Probability estimates: Enabled")
        print("  Class balancing: Enabled (handles imbalanced data)")
        print("  C parameter: 1.0 (regularization)")
        print("  Gamma: scale (kernel coefficient)")
        print("\n⚠️  NOTE: SVMs typically produce poorly calibrated probabilities!")
        print("  This makes them perfect candidates for calibration.")
        
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        )
        
        print("\nTraining in progress...")
        svm_model.fit(X_train_scaled, y_train)
        print("✓ Model training complete")
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("BASE MODEL EVALUATION")
        print("="*60)
        
        y_pred = svm_model.predict(X_test_scaled)
        y_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, svm_model.predict(X_train_scaled)),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1_score': f1_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_roc_auc': roc_auc_score(y_test, y_proba),
            'test_brier_score': brier_score_loss(y_test, y_proba)
        }
        
        print("\nBase Model Performance:")
        for metric, value in metrics.items():
            print(f"  {metric:.<40} {value:.4f}")
        
        mlflow.log_metrics(metrics)
        
        # Analyze probability distribution
        print("\n" + "="*60)
        print("PROBABILITY DISTRIBUTION ANALYSIS")
        print("="*60)
        
        print(f"\nPredicted Probabilities:")
        print(f"  Mean: {np.mean(y_proba):.4f}")
        print(f"  Median: {np.median(y_proba):.4f}")
        print(f"  Std Dev: {np.std(y_proba):.4f}")
        print(f"  Min: {np.min(y_proba):.4f}")
        print(f"  Max: {np.max(y_proba):.4f}")
        
        print(f"\nActual Default Rate: {np.mean(y_test):.4f}")
        print(f"Mean Prediction: {np.mean(y_proba):.4f}")
        print(f"Calibration Gap: {abs(np.mean(y_proba) - np.mean(y_test)):.4f}")
        
        # Check for extreme probabilities (sign of poor calibration)
        extreme_low = np.sum(y_proba < 0.1) / len(y_proba) * 100
        extreme_high = np.sum(y_proba > 0.9) / len(y_proba) * 100
        
        print(f"\nExtreme Predictions (sign of overconfidence):")
        print(f"  Probabilities < 0.1: {extreme_low:.1f}%")
        print(f"  Probabilities > 0.9: {extreme_high:.1f}%")
        
        if extreme_low + extreme_high > 50:
            print("\n Warning: Model shows signs of overconfidence!")
            print("  Many predictions are near 0 or 1.")
            print("  This is typical for SVMs and why calibration is crucial.")
        
        # Save the trained model
        if not os.path.exists('models/'): 
            os.makedirs("models/")
        
        model_filename = f'model_{timestamp}_base.joblib'
        dump(svm_model, model_filename)
        print(f"\n✓ Saved base model: {model_filename}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_filename': model_filename,
            'model_type': 'SVM_RBF',
            'dataset': 'Credit Card Default (UCI ID: 350)',
            'metrics': metrics,
            'n_features': X.shape[1],
            'default_rate': float(np.mean(y)),
            'class_balance': 'weighted',
            'scaling': 'StandardScaler',
            'kernel': 'rbf',
            'n_support_vectors': int(svm_model.n_support_.sum()),
            'ready_for_calibration': True
        }
        
        with open(f'model_{timestamp}_metadata.pickle', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Base SVM model ready for calibration")
        print(f"Model will be calibrated on {len(y_calib)} held-out samples")
        
        print(f"\nWhy SVMs Need Calibration:")
        print("  • SVMs optimize for classification, not probability estimation")
        print("  • Distance to hyperplane ≠ true probability")
        print("  • Tend to produce overconfident predictions (near 0 or 1)")
        print("  • Calibration transforms distances into reliable probabilities")
        
        print(f"\nWhy This Matters for Credit Risk:")
        print("  • Banks need accurate probabilities, not just classifications")
        print("  • Interest rates depend on precise default probability estimates")
        print("  • Regulatory compliance requires explainable risk scores")
        print("  • Portfolio management needs well-calibrated risk assessments")