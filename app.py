from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
model = load_model('finalmodel.h5')
print('Model loaded. Check http://127.0.0.1:5000/')



def model_predict(img_path, model):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    new_data=[]
    new_data.append(image)
    new_data = np.array(new_data) / 255.0
    

    #print(predictions)

    preds = model.predict(new_data)
    return preds


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
        preds = model_predict(file_path, model)
        classes=np.argmax(preds,axis=1)
        for ele in classes:
            if ele==1:
                result="COVID Negative"
            else:
                
                result="COVID Positive"

        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)

