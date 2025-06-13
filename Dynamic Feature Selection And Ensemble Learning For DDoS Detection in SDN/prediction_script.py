import argparse
import pandas as pd
import os
import glob
from ml_model import IntrusionDetectionModel

def predict_from_csv(model, input_file, output_file=None):
    """Make predictions for all rows in a CSV file"""
    print(f"Making predictions on {os.path.basename(input_file)}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"File contains {df.shape[0]} records with {df.shape[1]} features")
    
    # Check if there's a target column
    target_column = None
    possible_targets = ['Label', 'label', 'CLASS', 'class', 'target', 'Target', 'attack_type']
    for col in possible_targets:
        if col in df.columns:
            target_column = col
            break
    
    # Create a copy of the dataframe without the target column
    if target_column:
        print(f"Found target column: {target_column}")
        X = df.drop([target_column], axis=1)
        true_labels = df[target_column]
    else:
        print("No target column found")
        X = df
        true_labels = None
    
    # Make predictions in batches to avoid memory issues
    results = []
    batch_size = 1000
    batches = range(0, len(X), batch_size)
    
    for i in batches:
        batch_end = min(i + batch_size, len(X))
        print(f"Processing records {i+1} to {batch_end}...")
        
        batch_X = X.iloc[i:batch_end]
        
        # Process each row in the batch
        for idx, row in batch_X.iterrows():
            # Make prediction
            prediction = model.predict(row.to_dict())
            
            # Create result record
            result = {
                'record_id': idx,
                'predicted_class': prediction['prediction'],
                'attack_type': prediction['attack_type'],
                'confidence': prediction['confidence']
            }
            
            # Add true label if available
            if true_labels is not None:
                result['true_label'] = true_labels.iloc[idx]
            
            results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy if true labels are available
    if true_labels is not None:
        correct_predictions = sum(results_df['predicted_class'] == results_df['true_label'])
        accuracy = correct_predictions / len(results_df)
        print(f"Prediction accuracy: {accuracy:.4f}")
    
    # Save results if output file specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained models')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file or directory')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save prediction results')
    
    args = parser.parse_args()
    
    # Initialize the model and load trained models
    model = IntrusionDetectionModel()
    try:
        model.load_trained_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please train the models first")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single file
        output_file = os.path.join(
            args.output_dir, 
            f"predictions_{os.path.basename(args.input)}"
        )
        predict_from_csv(model, args.input, output_file)
    else:
        # Process all CSV files in directory
        csv_files = glob.glob(os.path.join(args.input, "*.csv"))
        if not csv_files:
            print(f"No CSV files found in {args.input}")
            return
        
        print(f"Found {len(csv_files)} CSV files")
        for csv_file in csv_files:
            output_file = os.path.join(
                args.output_dir, 
                f"predictions_{os.path.basename(csv_file)}"
            )
            predict_from_csv(model, csv_file, output_file)

if __name__ == "__main__":
    main()