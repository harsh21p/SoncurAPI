from flask import Flask, jsonify, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
import os

app = Flask(__name__)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

max_detections = 1

global category_index
directory_path = os.getcwd()

path = directory_path+"\Model"
labels_path = path+"\label_map.pbtxt"

print('Loading model...', end='')
detect_fn = tf.saved_model.load(path)
print('Done!')
category_index = label_map_util.create_category_index_from_labelmap(labels_path,use_display_name=True)

@app.route('/upload', methods = ['GET', 'POST'])
def home():
    global max_detections,category_index
    if(request.method == 'GET'):
        return render_template("home.html")
        

    if(request.method == 'POST'):

        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp

        f = request.files['file']

        image_np = load_image_into_numpy_array(f)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)

        scores = detections['detection_scores'][0, :max_detections].numpy()
        bboxes = detections['detection_boxes'][0, :max_detections].numpy()
        labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
        labels = [category_index[n]['name'] for n in labels]
        print(str(scores[0])+" "+str(labels[0]))
        if(scores[0]*100 >= 98):
            return "1,"+str(labels[0])
        
        return "0"

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000)
