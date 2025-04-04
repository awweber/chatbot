'''
Author: Your Name
Date: 2025-04-04

main.py
This script is the main entry point for a chatbot project using different ML methods for training.
'''
from chatbot_training import bow, vocabulary, le, decisionTree

# application imports
while True:
    userInput = input("Please enter a sentence [q: Exit]: ").lower()
    if userInput == "q":
        break
    userInput = [bow(vocabulary, userInput)]
    print(userInput)
    y_predict = decisionTree.predict(userInput)
    tag_predict = le.inverse_transform(y_predict)
    print(tag_predict)