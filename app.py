import streamlit as st
import numpy as np
import joblib
from pathlib import Path
# Import the custom preprocessing function designed to work with streamlit file uploader (byte arrays)
from load_data import load_data_for_prediction_from_array


# ---------------------------------
# Model Loading Function (Cached)
# ---------------------------------
# Use caching to load the model only once, speeding up the app.
@st.cache_resource
def load_trained_model(model_filename='svm_digit_model.joblib'):

    # Ensures path is correct regardless of where Streamlit is run from
    base_path = Path(__file__).parent
    model_path = base_path / model_filename

    try:
        loaded_model = joblib.load(model_path)
        return loaded_model
    except FileNotFoundError:
        st.error(f'❌ Error: Model file {model_filename} not found. Please ensure the model is in the correct path.')
        return None


# --------------------------
# Main Streamlit Function
# --------------------------
def main():
    # Set up the page config
    st.set_page_config(
        page_title='Persian Digit Recognition',
        page_icon='🔢',
        layout='centered'
    )

    # Main header
    st.title('🔢 Persian Digit Recognition (SVM)')
    st.markdown("""
    This application uses a **Support Vector Machine (SVM)** model to classify handwritten Persian digits (0-9). 
    Upload your image below for real-time prediction.
    """)
    st.divider()

    # Load Model
    model = load_trained_model()
    if model is None:
        return

    # File Uploader
    st.subheader('🖼️ Upload Your Handwritten Digit')
    uploaded_file = st.file_uploader('Choose a PNG, JPG, or JPEG image:', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:

        # Display Uploaded Image
        st.image(uploaded_file, caption='Uploaded Image', width=200)
        st.write('---')

        # Convert uploaded file to numpy byte array for cv2 decoding
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        # Prediction Section Header
        st.subheader('- - - - Prediction Results - - - -')

        # Call the preprocessing function which works directly with byte arrays
        processed_data = load_data_for_prediction_from_array(file_bytes)

        if processed_data is not None:
            # Predict
            prediction = model.predict(processed_data)

            # Display the processed 25x25 image that was fed into the model
            st.info('Model Input (25x25):')
            st.image(processed_data.reshape(25, 25).astype(np.uint8), width=100)

            # Display Final Result with a success banner
            st.success(f'### The model predicts the digit is: {prediction[0]} ✅')

            st.caption('Prediction is based on the pre-processed image shown above.')

        else:
            st.error('Image preprocessing failed. Please try a cleaner image. ❌')


if __name__ == "__main__":
    main()