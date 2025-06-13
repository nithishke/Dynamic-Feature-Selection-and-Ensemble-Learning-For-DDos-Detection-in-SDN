# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import os
import json
import pandas as pd
from ml_model import IntrusionDetectionModel

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.secret_key = os.urandom(24)

# Initialize the ML model
model = IntrusionDetectionModel()

# Load trained models if they exist
try:
    model.load_trained_models()
    print("Successfully loaded trained models")
except Exception as e:
    print(f"Warning: Could not load trained models. {e}")
    print("You need to train the models first by running train_model.py")

@app.route('/')
def home():
    """Render the landing page"""
    if 'user' in session:
        return render_template('home.html', logged_in=True)
    return render_template('landing.html', logged_in=False)

@app.route('/notebook')
def notebook():
    """Render the notebook page"""
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template('notebook.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html', logged_in='user' in session)

@app.route('/signup', methods=['POST'])
def signup():
    """Handle user sign-up"""
    data = request.form
    email = data.get('email')
    password = data.get('password')
    
    # For demonstration purposes, we'll use a simple JSON file as a "database"
    # In a production environment, use a proper database
    users_db_path = 'users.json'
    
    # Create users file if it doesn't exist
    if not os.path.exists(users_db_path):
        with open(users_db_path, 'w') as f:
            json.dump([], f)
    
    # Read existing users
    with open(users_db_path, 'r') as f:
        users = json.load(f)
    
    # Check if user already exists
    for user in users:
        if user['email'] == email:
            return jsonify({"status": "error", "message": "User already exists"}), 400
    
    # Add new user
    users.append({"email": email, "password": password})  # In production, hash the password!
    
    # Save updated users list
    with open(users_db_path, 'w') as f:
        json.dump(users, f)
    
    session['user'] = email
    return jsonify({"status": "success", "message": "User registered successfully"}), 200

@app.route('/login', methods=['POST'])
def login():
    """Handle user login"""
    data = request.form
    email = data.get('email')
    password = data.get('password')
    
    # Read users from "database"
    users_db_path = 'users.json'
    
    if not os.path.exists(users_db_path):
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401
    
    with open(users_db_path, 'r') as f:
        users = json.load(f)
    
    # Check credentials
    for user in users:
        if user['email'] == email and user['password'] == password:  # In production, verify the hashed password
            session['user'] = email
            return jsonify({"status": "success", "message": "Login successful"}), 200
    
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

@app.route('/logout')
def logout():
    """Handle user logout"""
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if 'user' not in session:
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert string values to appropriate types
        for key in data:
            try:
                data[key] = float(data[key])
            except ValueError:
                # Keep as string if conversion fails
                pass
        
        # Make prediction
        result = model.predict(data)
        
        return jsonify({
            "status": "success",
            "prediction": result["prediction"],
            "attack_type": result["attack_type"],
            "confidence": float(result["confidence"])
        }), 200
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/train', methods=['GET'])
def train_api():
    """API endpoint to trigger model training"""
    # This would typically be protected and only accessible by admins
    if 'user' not in session:
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    
    try:
        # For demonstration, we'll just return a message
        # In a real application, you might want to trigger a background job
        return jsonify({"status": "success", "message": "Training process would be triggered here"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)