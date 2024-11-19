#ONLY PREDICTION

from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model_path = "C:/Users/kashv/Downloads/VIT Downloads/Project/FER2013_TRAINED_MODEL"
model = load_model(model_path)

# Define categories (same as used during training)
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
IMG_SIZE = (48, 48)

# Preprocess and predict function
def predict_emotion(image_path, model, categories, img_size):
    img = imread(image_path, as_gray=True)
    img_resized = resize(img, img_size, anti_aliasing=True)
    img_array = img_resized.reshape(-1, img_size[0], img_size[1], 1).astype('float32') / 255.0
    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Example usage: predict emotion for a single image
image_path = "C:/Users/kashv/Downloads/VIT Downloads/Project/image1.jpg"  # Provide the path to your image
predicted_class, confidence = predict_emotion(image_path, model, CATEGORIES, IMG_SIZE)

print(f"Predicted emotion: {predicted_class}")

# Plotting the input image
img = imread(image_path)
plt.imshow(img)
plt.title(f"Predicted Emotion: {predicted_class} ({confidence:.2f})")
plt.axis('off')
plt.show()
