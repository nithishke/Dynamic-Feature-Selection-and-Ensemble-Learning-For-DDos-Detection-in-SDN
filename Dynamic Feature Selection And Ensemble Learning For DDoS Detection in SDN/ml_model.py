import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import glob

class IntrusionDetectionModel:
    def __init__(self):
        # Initialize classifiers with appropriate parameters
        self.models = {
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'passive_aggressive': PassiveAggressiveClassifier(random_state=42),
            'naive_bayes': GaussianNB()
            # Stacking will be initialized after other models
        }
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.pca = None
        self.selected_features = None
        self.model_directory = 'models'
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
    
    def process_multiple_csv_files(self, data_dir, sample_size=None):
        """Process multiple CSV files from a directory"""
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
        
        # Process each file and combine the data
        combined_X = None
        combined_y = None
        
        for csv_file in csv_files:
            print(f"Processing {os.path.basename(csv_file)}...")
            X, y = self.load_and_preprocess_data(csv_file, sample_size=sample_size)
            
            if combined_X is None:
                combined_X = X
                combined_y = y
            else:
                # Handle different columns between files
                # Ensure all columns from both dataframes are included
                all_columns = set(list(combined_X.columns) + list(X.columns))
                
                # Add missing columns with zeros
                for col in all_columns:
                    if col not in combined_X.columns:
                        combined_X[col] = 0
                    if col not in X.columns:
                        X[col] = 0
                
                # Align columns before concatenation
                X = X[combined_X.columns]
                
                # Concatenate the dataframes
                combined_X = pd.concat([combined_X, X], ignore_index=True)
                combined_y = pd.concat([combined_y, y], ignore_index=True)
        
        print(f"Combined dataset shape: {combined_X.shape}")
        return combined_X, combined_y
    
    def load_and_preprocess_data(self, data_path, sample_size=None):
        """Load and preprocess a single CSV file"""
        print(f"Loading data from {os.path.basename(data_path)}...")
        
        # Load the CSV file
        df = pd.read_csv(data_path)
        
        # Sample data if specified
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Find the target column
        target_column = self._find_target_column(df)
        if not target_column:
            raise ValueError(f"Could not find target column in {os.path.basename(data_path)}")
        
        # Extract features and target
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Convert categorical features to numeric using one-hot encoding
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols)
        
        # Convert label to numeric if needed
        if y.dtype == 'object':
            label_mapping = {label: idx for idx, label in enumerate(y.unique())}
            y = y.map(label_mapping)
            # Save the mapping
            joblib.dump(label_mapping, f"{self.model_directory}/label_mapping.pkl")
            print(f"Label mapping saved: {label_mapping}")
        
        return X, y
    
    def _find_target_column(self, df):
        """Find the target/label column in the dataset"""
        possible_target_columns = ['Label', 'label', 'CLASS', 'class', 'target', 'Target', 'attack_type']
        for col in possible_target_columns:
            if col in df.columns:
                return col
        return None
    
    def _handle_missing_values(self, X):
        """Handle missing values in the dataset"""
        # Fill numeric columns with mean
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        # Fill categorical columns with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        
        return X
    
    def feature_selection(self, X, y, k=10):
        """Feature selection using Mutual Information
        
        Mutual Information measures how much information a feature provides 
        about the target variable. Higher values indicate more informative features.
        """
        print("Performing feature selection with Mutual Information...")
        
        # Consider only numeric features for mutual information calculation
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            print("Warning: No numeric features found. Skipping feature selection.")
            self.selected_features = X.columns.tolist()
            return X
        
        try:
            # Calculate mutual information between each feature and the target
            mi_scores = mutual_info_classif(numeric_X, y, random_state=42)
            
            # Create a Series of features and their scores for easy sorting
            mi_series = pd.Series(mi_scores, index=numeric_X.columns)
            
            # Sort features by mutual information score (descending)
            sorted_features = mi_series.sort_values(ascending=False)
            
            # Print top features and their scores
            print("Top features by mutual information score:")
            for feature, score in sorted_features.head(k).items():
                print(f"  {feature}: {score:.4f}")
            
            # Select top k features (or all if less than k)
            k = min(k, len(sorted_features))
            selected_numeric_features = sorted_features.head(k).index.tolist()
            
            # Get categorical features (already one-hot encoded)
            categorical_features = list(set(X.columns) - set(numeric_X.columns))
            
            # Combine selected numeric and all categorical features
            self.selected_features = selected_numeric_features + categorical_features
            
            print(f"Selected {len(self.selected_features)} features in total")
            
            # Save the selected features list
            joblib.dump(self.selected_features, f"{self.model_directory}/selected_features.pkl")
            
            # Return the dataset with only selected features
            return X[self.selected_features]
            
        except Exception as e:
            print(f"Error in feature selection: {e}")
            print("Using all features instead")
            self.selected_features = X.columns.tolist()
            return X
    
    def apply_pca(self, X, n_components=0.95):
        """Apply PCA for dimensionality reduction
        
        PCA finds principal components (new features that are linear combinations
        of original features) that capture most of the variance in the data.
        """
        print("Applying PCA for dimensionality reduction...")
        
        try:
            # First, standardize the data (important for PCA)
            X_scaled = self.scaler.fit_transform(X)
            
            # Save the scaler for later use during prediction
            joblib.dump(self.scaler, f"{self.model_directory}/scaler.pkl")
            
            # Apply PCA
            # n_components=0.95 means retain enough components to explain 95% of variance
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X_scaled)
            
            # Print variance explained
            explained_variance = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            print(f"Original dimensions: {X.shape[1]}")
            print(f"PCA dimensions: {X_pca.shape[1]}")
            print(f"Total variance explained: {np.sum(explained_variance):.4f}")
            
            # Print first few components and their variance explained
            for i in range(min(5, len(explained_variance))):
                print(f"  Component {i+1}: explains {explained_variance[i]:.4f} of variance")
                
            # Save the PCA model
            joblib.dump(self.pca, f"{self.model_directory}/pca.pkl")
            
            return X_pca
            
        except Exception as e:
            print(f"Error in PCA: {e}")
            print("Skipping PCA, using standardized features instead")
            X_scaled = self.scaler.fit_transform(X)
            joblib.dump(self.scaler, f"{self.model_directory}/scaler.pkl")
            return X_scaled
    
    def train_models(self, X, y):
        """Train all models (MLP, Passive Aggressive, Naive Bayes, and Stacking)"""
        print("Training models...")
        
        # Initialize the stacking classifier 
        # This uses the three base models and combines their predictions using another model
        self.models['stacking'] = StackingClassifier(
            estimators=[
                ('mlp', self.models['mlp']),
                ('passive_aggressive', self.models['passive_aggressive']),
                ('naive_bayes', self.models['naive_bayes'])
            ],
            final_estimator=MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
        )
        
        # Train each model and save it
        for name, model in self.models.items():
            print(f"Training {name} model...")
            model.fit(X, y)
            
            # Save the model to disk
            model_path = f"{self.model_directory}/{name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
    
    def evaluate_models(self, X, y):
        """Evaluate all models and print performance metrics"""
        results = {}
        
        print("\nModel Evaluation Results:")
        print("=========================")
        
        for name, model in self.models.items():
            print(f"\n{name.upper()} MODEL:")
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate accuracy
            accuracy = accuracy_score(y, y_pred)
            
            # Generate classification report with precision, recall, and F1-score
            report = classification_report(y, y_pred)
            report_dict = classification_report(y, y_pred, output_dict=True)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'report': report_dict
            }
            
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print(report)
        
        # Save evaluation results
        joblib.dump(results, f"{self.model_directory}/evaluation_results.pkl")
        
        return results
    
    def load_trained_models(self):
        """Load all trained models from disk"""
        print("Loading trained models...")
        
        # Load feature selection
        features_path = f"{self.model_directory}/selected_features.pkl"
        if os.path.exists(features_path):
            self.selected_features = joblib.load(features_path)
            print(f"Loaded {len(self.selected_features)} selected features")
        
        # Load scaler
        scaler_path = f"{self.model_directory}/scaler.pkl"
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("Loaded scaler")
        
        # Load PCA
        pca_path = f"{self.model_directory}/pca.pkl"
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
            print(f"Loaded PCA with {self.pca.n_components_} components")
        
        # Load all models
        for name in self.models.keys():
            model_path = f"{self.model_directory}/{name}_model.pkl"
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"Loaded {name} model")
            else:
                print(f"Warning: {name} model file not found")
    
    def predict(self, input_data):
        """Make predictions using the stacking classifier"""
        try:
            # Convert input to DataFrame if it's not already
            if not isinstance(input_data, pd.DataFrame):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()
            
            # Handle missing values
            input_df = self._handle_missing_values(input_df)
            
            # Convert categorical features to one-hot encoding if needed
            categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                input_df = pd.get_dummies(input_df, columns=categorical_cols)
            
            # Ensure the input has all required features
            if self.selected_features:
                # Add missing columns with zeros
                for feature in self.selected_features:
                    if feature not in input_df.columns:
                        input_df[feature] = 0
                
                # Select only the features used during training
                input_df = input_df[self.selected_features]
            
            # Scale the data
            input_scaled = self.scaler.transform(input_df)
            
            # Apply PCA if it was used during training
            if self.pca is not None:
                input_transformed = self.pca.transform(input_scaled)
            else:
                input_transformed = input_scaled
            
            # Make prediction using the stacking classifier
            prediction = self.models['stacking'].predict(input_transformed)
            probabilities = self.models['stacking'].predict_proba(input_transformed)
            
            # Get the highest probability and its index
            max_prob_idx = np.argmax(probabilities[0])
            max_prob = probabilities[0][max_prob_idx]
            
            # Map prediction to attack type
            attack_type = self.get_attack_type(prediction[0])
            
            return {
                'prediction': prediction[0],
                'confidence': max_prob,
                'attack_type': attack_type
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                'prediction': None,
                'confidence': 0,
                'attack_type': "Error",
                'error_message': str(e)
            }
    
    def get_attack_type(self, label):
        """Map numeric label to attack type name"""
        # Try to load label mapping if available
        label_mapping_path = f"{self.model_directory}/label_mapping.pkl"
        if os.path.exists(label_mapping_path):
            label_mapping = joblib.load(label_mapping_path)
            # Reverse the mapping to get original label
            reverse_mapping = {v: k for k, v in label_mapping.items()}
            return reverse_mapping.get(label, f"Unknown ({label})")
        
        # Default mapping if no saved mapping exists
        attack_types = {
            0: "Normal Traffic",
            1: "DDoS Attack"
            # Add more mappings based on your dataset labels
        }
        
        return attack_types.get(label, f"Unknown Attack Type ({label})")