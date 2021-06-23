# 1. Install Dependencies
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("keras")
install("tensorflow")
install("Pillow")

import numpy as np
from PIL import Image
from keras.models import load_model
import boto3
import pickle
import argparse
import os 

class Score:
    """
    A sample Digit Recognizer Model handler implementation.
    """
    # Function to get prediction

    def predict(self, input):

        # Predict & Return
        return self.digit_model.predict_proba(input)[0, 1]

    # Function to get the model
    def load(self):

        model_path = load_model(os.path.join(os.environ['MODEL_PATH'], 'kerasDigitRecognizer.h5')) 
        model_path.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Loaded model from sagemaker")

        self.digit_model = model_path
        return True

if __name__ == "__main__":
    score = Score()
    score.load()
    img = np.array(Image.open(os.path.join(os.environ['DATA_PATH'], 'four.jpg')))
    img = img.reshape(-1, 28*28)
    # Preprocess - Normalization
    testInput_processed = img/ 255

    with open(os.path.join(os.environ['OUTPUT_PATH'], 'output.txt'), 'w') as output_fd:
        prediction = score.predict(testInput_processed)
        final_prediction = print ("Prediction for the input image is {}".format(prediction.argmax()))
        output_fd.write(str(final_prediction))
        output_fd.write('\n')