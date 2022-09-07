import tensorflow as tf
import numpy as np
import gradio as gr

# importing the model 
model = tf.keras.models.load_model('digit_classification.h5')


# A function to take an image and resize it, then classify it
def classify_image(image):
    img = image.reshape(-1, 28, 28)
    image_size = img / 255.0
    prediction = model.predict(image_size)
    return np.argmax(prediction)


# creating the interface
my_app = gr.Interface(
    classify_image, 
    'sketchpad', 
    'label', 
    live="True",
    title="Softmax Digit Classification"
)


# here I launch the app, so when we execute this file the app will be created
my_app.close()
my_app.launch(debug="True", share="True")