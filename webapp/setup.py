from flask import Flask, render_template, redirect, request, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

MODEL = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'efficientnetv2s.h5'))
REC_MODEL = pickle.load(open(os.path.join(MODEL_DIR, 'RF.pkl'), 'rb'))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASSES = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 'Cherry (including sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn(maize) Common rust', 'Corn(maize) Northern Leaf Blight', 'Corn(maize) healthy', 'Grape Black rot', 'Grape Esca(Black Measles)', 'Grape Leaf blight(Isariopsis Leaf Spot)', 'Grape healthy', 'Orange Haunglongbing(Citrus greening)', 'Peach Bacterial spot', 'Peach healthy', 'Bell PepperBacterial_spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites (Two-spotted spider mite)', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    image_folder = os.path.join('static', 'test_images')
    images = os.listdir(image_folder)
    images = ['test_images/' + file for file in images]
    return render_template("index.html", images=images)

@app.route('/plantdisease', methods=['GET', 'POST'])
def plantdisease():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            imagefile = tf.keras.utils.load_img(file_path, target_size=(224, 224))
            input_arr = tf.keras.preprocessing.image.img_to_array(imagefile)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            predict = MODEL.predict(input_arr)
            p = np.argmax(predict[0])
            res = CLASSES[p].replace('_', ' ')
            return redirect(url_for('plantresult', res=res, image_path=os.path.join('uploads', filename)))
    return render_template("plantdisease.html")

@app.route('/plantdisease/<res>/<image_path>')
def plantresult(res, image_path):
    return render_template('plantdiseaseresult.html', corrected_result=res, image_path=image_path)

@app.route('/croprecommendation/<res>')
def cropresult(res):
    print(res)
    corrected_result = res
    return render_template('croprecresult.html', corrected_result=corrected_result)

@app.route('/croprecommendation', methods=['GET', 'POST'])
def cr():
    if request.method == 'POST':
        X = []
        if request.form.get('nitrogen'):
            X.append(float(request.form.get('nitrogen')))
        if request.form.get('phosphorous'):
            X.append(float(request.form.get('phosphorous')))
        if request.form.get('potassium'):
            X.append(float(request.form.get('potassium')))
        if request.form.get('temperature'):
            X.append(float(request.form.get('temperature')))
        if request.form.get('humidity'):
            X.append(float(request.form.get('humidity')))
        if request.form.get('ph'):
            X.append(float(request.form.get('ph')))
        if request.form.get('rainfall'):
            X.append(float(request.form.get('rainfall')))
        X = np.array(X)
        X = X.reshape(1, -1)
        res = REC_MODEL.predict(X)[0]
        # print(res)
        return redirect(url_for('cropresult', res=res))
    return render_template('croprec.html')



if __name__== "__main__":
    app.run(debug=True)