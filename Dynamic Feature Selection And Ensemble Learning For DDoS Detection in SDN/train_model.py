import os
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train intrusion detection models')
parser.add_argument('--data_dir', type=str, default='CICDDOS2019', help='Directory containing dataset CSV files')
parser.add_argument('--output_dir', type=str, default='models', help='Directory to save trained models')
parser.add_argument('--max_samples', type=int, default=100000, help='Maximum number of samples to use for training')
parser.add_argument('--chunk_size', type=int, default=10000, help='Chunk size for reading large CSV files')
args = parser.parse_args()

def load_data_in_chunks(folder_path=args.data_dir, max_samples=args.max_samples, chunk_size=args.chunk_size):
    """Load data from CSV files in chunks to manage memory usage"""
    print(f"Loading data from folder: {folder_path} (max {max_samples} samples)")
    
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Initialize empty dataframe to collect samples
    sampled_data = []
    total_samples = 0
    
    # Process each file one by one
    for file in all_files:
        try:
            file_name = os.path.basename(file)
            print(f"Processing {file_name}")
            
            # Get number of rows in file without loading it completely
            with open(file, 'r') as f:
                num_lines = sum(1 for _ in f)
            print(f"File contains approximately {num_lines} rows")
            
            # Calculate sample ratio to get representative data from each file
            # We'll take max_samples / total_rows_all_files proportion from each file
            sample_ratio = min(0.1, max_samples / num_lines)  # Cap at 10% to ensure we sample from all files
            samples_from_file = int(num_lines * sample_ratio)
            
            # Read and sample the file in chunks
            chunk_reader = pd.read_csv(file, chunksize=chunk_size, low_memory=False)
            file_samples = 0
            
            for i, chunk in enumerate(chunk_reader):
                # Sample from this chunk based on the ratio
                chunk_sample_size = min(int(chunk.shape[0] * sample_ratio), chunk.shape[0])
                
                if chunk_sample_size > 0:
                    # Sample without replacement
                    chunk_sample = chunk.sample(n=chunk_sample_size, random_state=42+i)
                    sampled_data.append(chunk_sample)
                    file_samples += chunk_sample.shape[0]
                    total_samples += chunk_sample.shape[0]
                
                print(f"  Processed chunk {i+1}, samples from file: {file_samples}, total: {total_samples}")
                
                # Stop if we've collected enough samples
                if total_samples >= max_samples:
                    break
            
            print(f"Collected {file_samples} samples from {file_name}")
            
            # Stop if we've collected enough samples across all files
            if total_samples >= max_samples:
                print(f"Reached maximum sample count ({max_samples})")
                break
                
        except Exception as e:
            print(f"Failed to process {file}: {e}")
    
    # Combine all the sampled chunks
# Combine all the sampled chunks
    if sampled_data:
        combined_df = pd.concat(sampled_data, ignore_index=True)
    # Strip whitespace from column names
        combined_df.columns = combined_df.columns.str.strip()
        print(f"Final dataset shape: {combined_df.shape}")
        return combined_df
    else:
        print("No data loaded.")
        return None

def preprocess_data(df):
    print("Preprocessing data...")
    print("Available columns:", df.columns.tolist())
    
    # Handle infinity and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Find label column
    if 'Label' in df.columns:
        label_col = 'Label'
    else:
        # Try alternate label column names
        possible_label_columns = ['label', 'class', 'attack_type', ' Label']  # Added ' Label' with space
        for col in possible_label_columns:
            if col in df.columns:
                label_col = col
                break
        else:
            # If no label column is found
            print("Error: No Label column found in the dataset.")
            return None, None, None, None, None, None
    
    # Encode the labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"Label encoding: {label_mapping}")
    
    # Remove the label column and non-numeric columns
    X = df.drop(label_col, axis=1)
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    X = X.drop(non_numeric_cols, axis=1)
    
    # Clip extreme values
    for col in X.columns:
        X[col] = np.clip(X[col], a_min=-1e5, a_max=1e5)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, le, X.columns.tolist()

def select_features(X_train, X_test, y_train, n_features=15):
    print(f"Selecting top {n_features} features using Mutual Information...")
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create DataFrames for feature selection
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Select the most informative features
    selector = SelectKBest(mutual_info_classif, k=min(n_features, len(X_train.columns)))
    X_train_selected = selector.fit_transform(X_train_df, y_train)
    X_test_selected = selector.transform(X_test_df)
    
    # Get names of selected features
    selected_indices = selector.get_support(indices=True)
    selected_features = [X_train.columns[i] for i in selected_indices]
    
    print(f"Selected features: {selected_features}")
    
    return X_train_selected, X_test_selected, selector, selected_features

def apply_pca(X_train_selected, X_test_selected, n_components=10):
    print(f"Applying PCA to reduce dimensions to {n_components} components...")
    
    # Scale the data for PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Apply PCA
    n_components = min(n_components, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    return X_train_pca, X_test_pca, scaler, pca

def train_models(X_train, y_train):
    print("Training models...")
    models = {}
    
    # Train MLP model
    try:
        print("Training MLP...")
        mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42)
        mlp.fit(X_train, y_train)
        models['mlp'] = mlp
    except Exception as e:
        print(f"Error training MLP: {e}")
    
    # Train Passive Aggressive model
    try:
        print("Training Passive Aggressive...")
        pa = PassiveAggressiveClassifier(random_state=42)
        pa.fit(X_train, y_train)
        models['passive_aggressive'] = pa
    except Exception as e:
        print(f"Error training Passive Aggressive: {e}")
    
    # Train Naive Bayes model
    try:
        print("Training Naive Bayes...")
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        models['naive_bayes'] = nb
    except Exception as e:
        print(f"Error training Naive Bayes: {e}")
    
    # Train Stacking model if all base models were trained
    if len(models) >= 3:
        try:
            print("Training Stacking Classifier...")
            estimators = [
                ('mlp', MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42)),
                ('pa', PassiveAggressiveClassifier(random_state=42)),
                ('nb', GaussianNB())
            ]
            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=GaussianNB()
            )
            stacking_clf.fit(X_train, y_train)
            models['stacking'] = stacking_clf
        except Exception as e:
            print(f"Error training Stacking Classifier: {e}")
    
    print(f"Successfully trained {len(models)} models.")
    return models

def evaluate_models(models, X_test, y_test):
    print("Evaluating models...")
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'accuracy': accuracy,
            'report': report
        }
        
        print(f"\n{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

def save_models(models, selector, scaler, pca, le, selected_features, all_features, output_dir=args.output_dir):
    print(f"Saving models and preprocessing objects to {output_dir}...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save models
    for name, model in models.items():
        model_path = f'{output_dir}/{name}_model.pkl'
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} model to {model_path}")
        except Exception as e:
            print(f"Error saving {name} model: {e}")
    
    # Save preprocessing objects
    try:
        with open(f'{output_dir}/selector.pkl', 'wb') as f:
            pickle.dump(selector, f)
        
        with open(f'{output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        with open(f'{output_dir}/pca.pkl', 'wb') as f:
            pickle.dump(pca, f)
            
        with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
        
        # Save feature information and create model_results.json
        results = {}
        for name in models.keys():
            results[name] = {
                'model_path': f'{output_dir}/{name}_model.pkl',
                'n_features': len(selected_features),
                'n_components': pca.n_components_
            }
        
        import json
        with open(f'{output_dir}/model_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("All preprocessing objects and model results saved.")
    except Exception as e:
        print(f"Error saving preprocessing objects: {e}")

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data in memory-efficient chunks
    df = load_data_in_chunks(max_samples=args.max_samples, chunk_size=args.chunk_size)
    if df is None:
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, le, all_features = preprocess_data(df)
    if X_train is None:
        return
    
    # Free up memory
    del df
    
    # Select features
    X_train_selected, X_test_selected, selector, selected_features = select_features(X_train, X_test, y_train)
    
    # Free up memory
    del X_train, X_test
    
    # Apply PCA
    X_train_pca, X_test_pca, scaler, pca = apply_pca(X_train_selected, X_test_selected)
    
    # Free up memory
    del X_train_selected, X_test_selected
    
    # Train models
    models = train_models(X_train_pca, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test_pca, y_test)
    
    # Save models and preprocessing objects
    save_models(models, selector, scaler, pca, le, selected_features, all_features)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()