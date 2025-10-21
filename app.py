"""
FastAPI application for Credit Card Default Prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import joblib
import pickle
import numpy as np
import glob
from contextlib import asynccontextmanager

# Global variables for model
model = None
scaler = None
metadata = None

def load_model():
    """Load the latest calibrated model"""
    global model, scaler, metadata
    
    # Find latest calibrated model
    model_files = glob.glob('models/model_*_calibrated_sigmoid.joblib')
    if not model_files:
        raise FileNotFoundError("No calibrated model found!")
    
    latest_model = sorted(model_files)[-1]
    timestamp = latest_model.split('_')[1]
    
    model = joblib.load(latest_model)
    scaler = joblib.load('data/scaler.pickle')
    
    metadata_file = f'models/model_{timestamp}_metadata.pickle'
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✓ Loaded model with timestamp: {timestamp}")
    print(f"  Model type: {metadata['model_type']}")
    print(f"  Dataset: {metadata['dataset']}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    print("\n" + "="*60)
    print("Starting Credit Card Default Prediction API")
    print("="*60)
    load_model()
    print("✓ Model loaded successfully")
    print("="*60 + "\n")
    yield
    print("\nShutting down API...")

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Default Prediction API",
    description="API for predicting credit card default probability using calibrated SVM",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        description="List of 23 customer features",
        min_items=23,
        max_items=23
    )
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 23:
            raise ValueError('Must provide exactly 23 features')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [20000, 2, 2, 1, 24, 26, 0, 0, 0, 0, 0, 689, 0, 0, 0, 689, 0, 0, 0, 0, 0, 0, 0]
            }
        }

class Customer(BaseModel):
    customer_id: str = Field(None, description="Optional customer identifier")
    features: List[float] = Field(..., min_items=23, max_items=23)

class BatchPredictionRequest(BaseModel):
    customers: List[Customer]
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "customer_id": "CUST001",
                        "features": [20000, 2, 2, 1, 24, 26, 0, 0, 0, 0, 0, 689, 0, 0, 0, 689, 0, 0, 0, 0, 0, 0, 0]
                    },
                    {
                        "customer_id": "CUST002",
                        "features": [50000, 1, 2, 1, 30, 35, 0, 0, 0, 0, 0, 1200, 1100, 1000, 1000, 1000, 100, 200, 300, 500, 300, 500, 1000]
                    }
                ]
            }
        }

class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Probability of default (0-1)")
    default_probability_pct: str = Field(..., description="Probability as percentage")
    prediction: int = Field(..., description="Binary prediction (0=No Default, 1=Default)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    risk_level: str = Field(..., description="Risk category")
    recommendation: str = Field(..., description="Lending recommendation")
    suggested_interest_rate: str = Field(..., description="Suggested interest rate")

class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[dict]

class ModelInfo(BaseModel):
    model_type: str
    dataset: str
    n_features: int
    timestamp: str
    calibration_method: str = "Platt Scaling (Sigmoid)"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str = None

def interpret_risk(probability: float) -> dict:
    """Interpret default probability into risk categories"""
    if probability < 0.2:
        risk_level = "LOW RISK"
        recommendation = "Approve with standard terms"
        interest_rate = "Prime rate + 2%"
    elif probability < 0.4:
        risk_level = "MODERATE RISK"
        recommendation = "Approve with monitoring"
        interest_rate = "Prime rate + 5%"
    elif probability < 0.6:
        risk_level = "HIGH RISK"
        recommendation = "Approve with strict terms"
        interest_rate = "Prime rate + 10%"
    else:
        risk_level = "VERY HIGH RISK"
        recommendation = "Decline or require collateral"
        interest_rate = "Not recommended"
    
    return {
        'risk_level': risk_level,
        'recommendation': recommendation,
        'suggested_interest_rate': interest_rate
    }

# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Welcome message with API information"""
    return {
        "message": "Credit Card Default Prediction API",
        "version": "1.0.0",
        "model_type": metadata['model_type'] if metadata else "Not loaded",
        "docs": "/docs",
        "endpoints": {
            "GET /health": "Check API health",
            "GET /model_info": "Get model information",
            "POST /predict": "Single customer prediction",
            "POST /batch_predict": "Batch predictions"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": metadata['model_type'] if metadata else None
    }

@app.get("/model_info", response_model=ModelInfo, tags=["General"])
async def get_model_info():
    """Get information about the loaded model"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": metadata['model_type'],
        "dataset": metadata['dataset'],
        "n_features": metadata['n_features'],
        "timestamp": metadata['timestamp']
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Predict default probability for a single customer
    
    **Features (23 total):**
    1. LIMIT_BAL: Credit limit
    2. SEX: Gender (1=male, 2=female)
    3. EDUCATION: Education level
    4. MARRIAGE: Marital status
    5. AGE: Age in years
    6-11. PAY_0 to PAY_6: Payment status (past 6 months)
    12-17. BILL_AMT1 to BILL_AMT6: Bill amounts (past 6 months)
    18-23. PAY_AMT1 to PAY_AMT6: Payment amounts (past 6 months)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Predict
        probability = float(model.predict_proba(features_scaled)[0, 1])
        prediction = int(model.predict(features_scaled)[0])
        
        # Interpret risk
        risk_info = interpret_risk(probability)
        
        return {
            "default_probability": probability,
            "default_probability_pct": f"{probability:.2%}",
            "prediction": prediction,
            "prediction_label": "Default" if prediction == 1 else "No Default",
            **risk_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict default probability for multiple customers
    
    Returns predictions for all customers in the batch
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        
        for idx, customer in enumerate(request.customers):
            # Prepare features
            features = np.array(customer.features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            # Predict
            probability = float(model.predict_proba(features_scaled)[0, 1])
            prediction = int(model.predict(features_scaled)[0])
            
            # Interpret risk
            risk_info = interpret_risk(probability)
            
            result = {
                "customer_id": customer.customer_id or f"customer_{idx}",
                "default_probability": probability,
                "default_probability_pct": f"{probability:.2%}",
                "prediction": prediction,
                "prediction_label": "Default" if prediction == 1 else "No Default",
                **risk_info
            }
            
            results.append(result)
        
        return {
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/calibration_report", tags=["Model Information"])
async def get_calibration_report():
    """Get the calibration report for the loaded model"""
    try:
        # Find latest calibration report
        report_files = glob.glob('models/calibration_report_*.pickle')
        if not report_files:
            raise HTTPException(status_code=404, detail="No calibration report found")
        
        latest_report = sorted(report_files)[-1]
        with open(latest_report, 'rb') as f:
            report = pickle.load(f)
        
        return {
            "calibration_method": report['calibration_method'],
            "timestamp": report['timestamp'],
            "base_metrics": {
                "brier_score": report['base_metrics']['brier_score'],
                "expected_calibration_error": report['base_metrics']['expected_calibration_error'],
                "roc_auc": report['base_metrics']['roc_auc'],
                "f1_score": report['base_metrics']['f1_score']
            },
            "calibrated_metrics": {
                "brier_score": report['calibrated_metrics']['brier_score'],
                "expected_calibration_error": report['calibrated_metrics']['expected_calibration_error'],
                "roc_auc": report['calibrated_metrics']['roc_auc'],
                "f1_score": report['calibrated_metrics']['f1_score']
            },
            "improvements": report['improvements']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load report: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting FastAPI Server")
    print("="*60)
    print("API will be available at:")
    print("  - Main API: http://localhost:8000")
    print("  - Interactive Docs: http://localhost:8000/docs")
    print("  - Alternative Docs: http://localhost:8000/redoc")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")