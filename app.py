import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
from skimage import io, transform

app = Flask(__name__)

model = joblib.load('model.pkl')

# @app.route('/', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             uploaded_image = request.files['image']

#             if uploaded_image.filename != '':
#                 temp_image_path = 'temp_image.jpg'
#                 uploaded_image.save(temp_image_path)

#                 input_image = preprocess_image(temp_image_path)

#                 predicted_label = model.predict(input_image)

#                 predicted_value = predicted_label[0]

#                 return render_template("predict.html")

#         except Exception as e:
#             return f'Error: {str(e)}'

#     return render_template('index.html')
# from flask import render_template  # Add render_template import

@app.route('/', methods=['GET', 'POST'])
def predict():
    predicted_value = None  # Initialize predicted_value variable
    
    if request.method == 'POST':
        try:
            uploaded_image = request.files['image']

            if uploaded_image.filename != '':
                temp_image_path = 'temp_image.jpg'
                uploaded_image.save(temp_image_path)

                input_image = preprocess_image(temp_image_path)

                predicted_label = model.predict(input_image)

                predicted_value = predicted_label[0]

                # Render the predict.html template with the predicted value
                return render_template("predict.html", predicted_value=predicted_value)

        except Exception as e:
            return f'Error: {str(e)}'

    return render_template('index.html')


def preprocess_image(image_path):
    img = io.imread(image_path)
    img = transform.resize(img, (256, 256))
    img = img.reshape(1, -1)
    return img

if __name__ == '__main__':
    app.run(debug=True, use_reloader = True)

