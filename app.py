from flask import Flask, jsonify, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
import os

app = Flask(__name__)

@app.route('/upload', methods = ['GET', 'POST'])
def home():
    return render_template("home.html")
     
if __name__ == '__main__':
   
    app.run(host="0.0.0.0",port=5000)
