#!/usr/bin/env python3

import json
import pickle
import numpy as np

def load_public_cases():
    """Load public test cases."""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    return data

def predict_xgboost(trip_duration_days, miles_traveled, total_receipts_amount):
    """Predict using XGBoost model."""
    try:
        with open('xgboost_deep_model.pkl', 'rb') as f:
            model = pickle.load(f)
        features = np.array([[trip_duration_days, miles_traveled, total_receipts_amount]])
        prediction = model.predict(features)[0]
        return round(prediction, 2)
    except:
        return None

def predict_linear(trip_duration_days, miles_traveled, total_receipts_amount):
    """Simple linear prediction for comparison."""
    return 119 * trip_duration_days + 0.5 * miles_traveled + 0.4 * total_receipts_amount - 20

def evaluate_model(predictions, actual_values, model_name):
    """Evaluate model performance."""
    if not predictions or len(predictions) != len(actual_values):
        print(f"{model_name}: Could not evaluate (missing predictions)")
        return
    
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    
    # Calculate metrics
    errors = predictions - actual_values
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    
    # Perfect matches
    perfect_matches = np.sum(np.abs(errors) < 0.01)
    near_perfect = np.sum(np.abs(errors) < 1.0)
    within_5 = np.sum(np.abs(errors) < 5.0)
    within_10 = np.sum(np.abs(errors) < 10.0)
    
    total = len(actual_values)
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  Perfect matches (±$0.01): {perfect_matches}/{total} ({100*perfect_matches/total:.1f}%)")
    print(f"  Near perfect (±$1.00): {near_perfect}/{total} ({100*near_perfect/total:.1f}%)")
    print(f"  Within $5.00: {within_5}/{total} ({100*within_5/total:.1f}%)")
    print(f"  Within $10.00: {within_10}/{total} ({100*within_10/total:.1f}%)")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'perfect_matches': perfect_matches,
        'near_perfect': near_perfect,
        'within_5': within_5,
        'within_10': within_10
    }

def main():
    print("Loading test cases...")
    test_cases = load_public_cases()
    print(f"Loaded {len(test_cases)} test cases")
    
    # Extract features and targets
    features = []
    targets = []
    
    for case in test_cases:
        features.append([
            case['input']['trip_duration_days'],
            case['input']['miles_traveled'],
            case['input']['total_receipts_amount']
        ])
        targets.append(case['expected_output'])
    
    print("\nGenerating predictions...")
    
    # XGBoost predictions
    xgboost_predictions = []
    for feat in features:
        pred = predict_xgboost(feat[0], feat[1], feat[2])
        xgboost_predictions.append(pred)
    
    # Linear predictions
    linear_predictions = []
    for feat in features:
        pred = predict_linear(feat[0], feat[1], feat[2])
        linear_predictions.append(pred)
    
    # Evaluate models
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    xgb_results = evaluate_model(xgboost_predictions, targets, "XGBoost (Deep)")
    linear_results = evaluate_model(linear_predictions, targets, "Linear Regression")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if xgb_results and linear_results:
        print(f"XGBoost RMSE: ${xgb_results['rmse']:.2f}")
        print(f"Linear RMSE: ${linear_results['rmse']:.2f}")
        print(f"XGBoost Perfect matches: {xgb_results['perfect_matches']}")
        print(f"Linear Perfect matches: {linear_results['perfect_matches']}")
        
        improvement = ((linear_results['rmse'] - xgb_results['rmse']) / linear_results['rmse']) * 100
        print(f"XGBoost RMSE improvement: {improvement:.1f}%")

if __name__ == "__main__":
    main() 