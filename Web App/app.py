from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model100.h5'

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img100 = cv2.imread(img_path)
    d = {0: 'NEUTROPHIL', 1: 'MONOCYTE', 2: 'LYMPHOCYTE', 3: 'EOSINOPHIL'}
    if img100 is not None:
        img100 = cv2.resize(img100, (60, 80))
        p=np.asarray(img100)
        f=[]
        f.append(p)
        f=np.asarray(f)
        fpt=model.predict(f)
        fp=np.argmax(fpt, axis=1)
        for i in fp:
            return d[i]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

