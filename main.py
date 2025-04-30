'''
Author: Your Name
Date: 2025-04-04

main.py
This script is the main entry point for a chatbot project using different ML methods for training.
'''

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# Load TensorFlow model and its dependencies
tf_model = load_model('tf_model/chatbot_tfmodelSeq.keras')
tf_count_vec = joblib.load('tf_model/count_vectorizer.pkl')
tf_le = joblib.load('tf_model/label_encoder.pkl')

# Load Decision Tree model and its dependencies
dt_model = joblib.load('decision_tree_model/decision_tree_model.pkl')
dt_count_vec = joblib.load('decision_tree_model/count_vectorizer.pkl')
dt_le = joblib.load('decision_tree_model/label_encoder.pkl')

def chatbot_interaction():
    """Handles the chatbot interaction with the user."""
    print("\nAvailable models:")
    print("1. TensorFlow Model")
    print("2. Decision Tree Model")
    
    while True:
        model_choice = input("Choose your model (1 or 2, q to quit): ").lower()
        if model_choice == 'q':
            print("Goodbye!")
            break
            
        if model_choice not in ['1', '2']:
            print("Invalid choice. Please choose 1 or 2.")
            continue
            
        active_model = 'tf' if model_choice == '1' else 'decisionTree'
        print(f"\nUsing {active_model} model. Type 'q' to quit or 'c' to change model.")
        
        while True:
            user_input = input("\nEnter your message [or 'q' to exit]: ").lower()
            if user_input == 'q':
                print("Goodbye!")
                return
            if user_input == 'c':
                print("\nChanging model...")
                break
                
            # Process input based on selected model
            if active_model == 'tf':
                # Transform input using TF vectorizer
                user_input_vec = tf_count_vec.transform([user_input]).toarray()
                prediction = tf_model.predict(user_input_vec)
                predicted_class = np.argmax(prediction, axis=1)
                tag_predict = tf_le.inverse_transform(predicted_class)
            else:
                # Transform input using DT vectorizer
                user_input_vec = dt_count_vec.transform([user_input]).toarray()
                prediction = dt_model.predict(user_input_vec)
                tag_predict = dt_le.inverse_transform(prediction)
                
            print(f"Predicted category: {tag_predict[0]}")

def main():
    """Main entry point for the chatbot application."""
    print("Welcome to the Chatbot!")
    print("Available categories:", ", ".join(tf_le.classes_))
    chatbot_interaction()

if __name__ == "__main__":
    main()