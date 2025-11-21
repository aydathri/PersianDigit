import numpy as np
import cv2
from scipy import io



# ------------------
# Dataset Loading
# ------------------
def load_dataset(training_size, test_size, resize):
    dataset = io.loadmat('dataset/Data_hoda_full.mat')

    data = np.squeeze(dataset['Data'])
    labels = np.squeeze(dataset['labels'])

    # Split data into training and testing sets based on specified sizes
    X_train = data[:training_size]
    y_train = labels[:training_size]
    X_test = data[training_size:training_size + test_size]
    y_test = labels[training_size:training_size + test_size]

    # Resize all images in both sets to the target
    X_train_resize = np.array([cv2.resize(img, (resize, resize)) for img in X_train])
    X_test_resize = np.array([cv2.resize(img, (resize, resize)) for img in X_test])

    # Flatten the 2D images into 1D vectors for model input
    reshape_size = resize * resize
    X_train = X_train_resize.reshape(-1, reshape_size)
    X_test = X_test_resize.reshape(-1, reshape_size)

    return X_train, y_train, X_test, y_test

# Use
# X_train, y_train, X_test, y_test = load_data(1000, 200, 22)
# plt.imshow(X_train[20].reshape(22,22), cmap='gray')
# plt.show()



# ---------------------------------
# Prediction from Local File Path
# ---------------------------------
def load_data_for_test_prediction(img_path, target_size=25):
    # Read the image from the file path in grayscale mode
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Error: Could not read image at {img_path}')
        return None

    # Apply inverse binary thresholding to clean the image (digit white, background black)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find external contours to isolate the digit area
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print('Error: No contour (digit) found in the image.')
        return None

    # Find the largest contour (assumed to be the handwritten digit)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Define padding and crop the image dynamically to center the digit
    padding = 5
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(binary_img.shape[1], x + w + padding)
    y_end = min(binary_img.shape[0], y + h + padding)

    cropped_img = binary_img[y_start:y_end, x_start:x_end]

    # Resize the cropped image to the model's required input size (e.g., 25x25)
    resized_img = cv2.resize(cropped_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Flatten the image into a 1D vector
    reshape_size = target_size * target_size
    final_input = resized_img.flatten().reshape(1, reshape_size)

    # Comment out the visual debugging lines for final usage
    # cv2.imshow('Final Input to SVM', resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return final_input



# ------------------------------------------
# Prediction from Byte Array for Streamlit
# ------------------------------------------
def load_data_for_prediction_from_array(img_array, target_size=25):
    # Processes image from numpy byte array (Streamlit input) using OpenCV.

    # Decode byte array into an OpenCV image (grayscale)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 1.Thresholding (using THRESH_BINARY_INV, as previously determined)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # 2.Find Contours for Smart Cropping/Centering
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 3.Crop with Padding
    padding = 5
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(binary_img.shape[1], x + w + padding)
    y_end = min(binary_img.shape[0], y + h + padding)

    cropped_img = binary_img[y_start:y_end, x_start:x_end]

    # 4.Resize to 25x25
    resized_img = cv2.resize(cropped_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 5.Flatten to 1x625 vector
    reshape_size = target_size * target_size
    final_input = resized_img.flatten().reshape(1, reshape_size)

    return final_input