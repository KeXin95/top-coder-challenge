#!/usr/bin/env python3

import pickle
import sys
import numpy as np

def predict_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Predict reimbursement amount using trained XGBoost model.
    
    Args:
        trip_duration_days: Number of days for the trip
        miles_traveled: Total miles traveled
        total_receipts_amount: Total amount in receipts
    
    Returns:
        Predicted reimbursement amount
    """
    try:
        # Load the trained model
        with open('xgboost_deep_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Prepare input features
        features = np.array([[trip_duration_days, miles_traveled, total_receipts_amount]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return round(prediction, 2)
        
    except FileNotFoundError:
        print("Error: XGBoost model file not found. Please run xgboost_analysis.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 4:
        print("Usage: python predict_xgboost.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
        
        result = predict_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
        print(result)
        
    except ValueError:
        print("Error: All arguments must be numbers")
        sys.exit(1)

if __name__ == "__main__":
    main() 