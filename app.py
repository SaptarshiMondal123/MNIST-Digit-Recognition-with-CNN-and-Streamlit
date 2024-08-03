import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import ImageOps, Image

# Function to load the model with error handling
def load_model():
    try:
        model = tf.keras.models.load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors: background should be black
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image).astype('float32') / 255
    image = np.expand_dims(image, axis=(0, -1))  # Ensure the image shape is (1, 28, 28, 1)
    return image

# Function to make predictions
def recognize(image):
    if image is not None:
        preprocessed_image = preprocess_image(image)
        st.image(preprocessed_image.squeeze(), caption='Preprocessed Image', use_column_width=True, clamp=True)

        prediction = model.predict(preprocessed_image)
        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return {}

# Streamlit app
st.title("MNIST Digit Recognizer")

st.write("Draw a digit in the box below:")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="white",  # Drawing background color
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    update_streamlit=True,
    height=300,  # Increased height for better drawing
    width=300,  # Increased width for better drawing
    drawing_mode="freedraw",
    key="canvas",
)

# Process the canvas result
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
