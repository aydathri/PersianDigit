import joblib
from load_data import load_data_for_test_prediction

try:
    # Load model
    model_filename = 'svm_digit_model.joblib'
    loaded_model = joblib.load(model_filename)
    print(f'Model {model_filename} loaded successfully.')
except FileNotFoundError:
    print(f'Error: Model file {model_filename} not found. Please train and save the model first.')
    exit()


image_path = 'digits_photo/digit_0.jpg'

processed_image = load_data_for_test_prediction(image_path)

if processed_image is not None:
    prediction = loaded_model.predict(processed_image)

    print(f'\n Image processed successfully.')
    print(f'Predicted Persian Digit: {prediction[0]}')
else:
    print(f'\n Prediction failed due to image processing error.')