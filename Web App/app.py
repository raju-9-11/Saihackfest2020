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
MODEL_PATH = 'model/model100.h5'

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
        result=preds
        if preds=="EOSINOPHIL":
            result=result+": Eosinophils, sometimes called eosinophiles or, less commonly, acidophils, are a variety of white blood cells and one of the immune system components responsible for combating multicellular parasites and certain infections in vertebrates."
        elif preds=="LYMPHOCYTE":
            result=result+":A lymphocyte is a type of white blood cell in the vertebrate immune system. Lymphocytes include natural killer cells (which function in cell-mediated, cytotoxic innate immunity), T cells (for cell-mediated, cytotoxic adaptive immunity), and B cells (for humoral, antibody-driven adaptive immunity)."
        elif preds=="MONOCYTE":
            result=result+":Monocytes are a type of leukocyte, or white blood cell. They are the largest type of leukocyte and can differentiate into macrophages and myeloid lineage dendritic cells. As a part of the vertebrate innate immune system monocytes also influence the process of adaptive immunity."
        else:
            result=result+":Neutrophils are a type of white blood cell. In fact, most of the white blood cells that lead the immune system's response are neutrophils. There are four other types of white blood cells. Neutrophils are the most plentiful type, making up 55 to 70 percent of your white blood cells."
        return result
        
    return None


if __name__ == '__main__':
    app.run(debug=True)

