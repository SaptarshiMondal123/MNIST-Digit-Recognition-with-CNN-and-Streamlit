import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import ImageOps, Image

def load_model():
    try:
        model = tf.keras.models.load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)  
    image = image.resize((28, 28))  
    image = np.array(image).astype('float32') / 255
    image = np.expand_dims(image, axis=(0, -1))  
    return image

def recognize(image):
    if image is not None:
        preprocessed_image = preprocess_image(image)
        st.image(preprocessed_image.squeeze(), caption='Preprocessed Image', use_column_width=True, clamp=True)

        prediction = model.predict(preprocessed_image)
        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return {}

st.title("MNIST Digit Recognizer")

st.write("Draw a digit in the box below:")

canvas_result = st_canvas(
    fill_color="white", 
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    update_streamlit=True,
    height=300,  
    width=300,  
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = Image.fromarray((img[:, :, :3] * 255).astype('uint8'), 'RGB')
    st.image(img, caption='Drawn Image.', use_column_width=True)

    if model is not None:
        st.write("Classifying...")
        predictions = recognize(img)
        st.write(predictions)
    else:
        st.error("Model not loaded. Please check the model file.")
