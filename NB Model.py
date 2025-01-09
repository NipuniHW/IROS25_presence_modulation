# MIT License
# 
# Copyright (c) [2024] Modulate Presence
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import joblib  
import os

# Load the dataset (data.csv)
data = pd.read_csv(Add the dataset path)

def NB(new_data):
    # Extract features and labels
    X = data[['ambient_confidence', 'keyword_confidence', 'sentiment_confidence']]
    y = data['label']

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Naive Bayes classifier
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Test the classifier
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Save the model using joblib
    model_save_path = Add the path to save the NB model
    joblib.dump(nb, model_save_path)
    print(f"Model saved as '{os.path.basename(model_save_path)}'")

    # Predict using the Naive Bayes model on new data
    prediction = nb.predict(new_data)

    # Interpret the result
    if prediction == 0:
        print("Alarmed, Emergency situation detected!")
    elif prediction == 1:
        print("Be Social")
    else:
        print("Stay Silent")

    # Wait for a few seconds before running again
    time.sleep(3)

if __name__ == '__main__':

    new_data = [[0.3, 0, 0.99]] 
    NB(new_data)
