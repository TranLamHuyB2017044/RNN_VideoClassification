from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from flask_cors import CORS


import joblib
app = Flask(__name__)
CORS(app)



def get_frames(file_path):
    _images_per_file = 20
    img_size = 224
    
    images = []
    
    vidcap = cv2.VideoCapture(file_path)
    success, image = vidcap.read()
    count = 0

    while success and count < _images_per_file:
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = cv2.resize(rgb_img, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
        images.append(res)

        success, image = vidcap.read()
        count += 1

    result = np.array(images)
    result = (result / 255.).astype(np.float16)

    return result


def get_transfer_values(file_name):
    _images_per_file = 20
    img_size = 224
    img_size_touple = (img_size, img_size)

    image_model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = image_model.get_layer('fc2')
    image_model_transfer = Model(inputs=image_model.input,
                                outputs=transfer_layer.output)
    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    
    shape = (_images_per_file,) + img_size_touple + (3,)
    
    image_batch = np.zeros(shape=shape, dtype=np.float16)

    image_batch = get_frames(file_name)

    shape = (_images_per_file, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    transfer_values = image_model_transfer.predict(image_batch)

    return transfer_values

def classify_video(video_file, model):
    class_names = {0: 'shoot', 1: 'heading', 2: 'keeper'}

    if not (video_file.filename.endswith('.mp4') or video_file.filename.endswith('.avi')):
        print("Định dạng video không hợp lệ. Chỉ chấp nhận định dạng .mp4 hoặc .avi.")
        return

    video_path = video_file.filename
    video_file.save(video_path)

    video_transfer_values = get_transfer_values(video_path)

    reshaped_transfer_values = video_transfer_values.reshape(1, 20, 4096)

    predictions = model.predict(reshaped_transfer_values)
    predicted_class_index = np.argmax(predictions)

    predicted_class = class_names.get(predicted_class_index, "Unknown")

    return predicted_class


@app.route('/predict', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    video_file = request.files['video']
    if video_file.name == '':
        return jsonify({'error': 'No selected file'}), 400

    if video_file:
        
        model_path = './model/model_cua_van_2.pkl'
        model = joblib.load(model_path)
        
        predicted_class = classify_video(video_file, model)  
        return jsonify({'data': predicted_class}), 200



if __name__ == '__main__':
    app.run(debug=True)
