'''
Author: Your Name
Date: 2025-04-04

main.py
This script is the main entry point for a chatbot project using different ML methods for training.
'''

from chatbot_decisionTreeClassifier import bow, vocabulary, le, decisionTree
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

chatbot_model = load_model('chatbot_tfmodelSeq.keras')

# active_model = 'tf'  # Set the active model to 'tf' or 'decisionTree'
active_model = 'decisionTree'  # Uncomment this line to use the Decision Tree model
labels = le.classes_  # Get the labels from the LabelEncoder
# ['goodbye' 'greeting' 'help' 'joke' 'name' 'purpose' 'thanks']

def chatbot_interaction(active_model='decisionTree'):
    """Handles the chatbot interaction with the user."""
    while True:
        user_input = input("Please enter a sentence [q: Exit]: ").lower()
        if user_input == "q":
            print("Exiting chatbot. Goodbye!")
            break
        user_input_bow = [bow(vocabulary, user_input)]
        # print(f"Bag of Words Representation: {user_input_bow}")
        count_of_ones = sum(user_input_bow[0])
        print(f"Number of 1s in Bag of Words: {count_of_ones}")
        if active_model == 'tf':
            print("Using TensorFlow model for prediction...")
            # Ensure user_input_bow is a NumPy array before converting to tensor
            user_input_bow = tf.convert_to_tensor(user_input_bow, dtype=tf.float32)
            prediction = chatbot_model.predict(user_input_bow)
            predicted_class = tf.argmax(prediction, axis=1).numpy()
            tag_predict = le.inverse_transform(predicted_class)
        elif active_model == 'decisionTree':
            print("Using Decision Tree model for prediction...")
            # Ensure user_input_bow is a NumPy array before converting to tensor
            y_predict = decisionTree.predict(user_input_bow)
            tag_predict = le.inverse_transform(y_predict)
        print(f"Predicted Tag: {tag_predict}")

def main():
    """Main entry point for the chatbot application."""
    print("Welcome to the Chatbot!")
    print("Chatbot is able to label your input to the following categories:\n", labels)
    chatbot_interaction(active_model=active_model)

if __name__ == "__main__":
    main()