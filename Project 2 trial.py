# Updated Code for Training and Prediction

from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define paths
TRAIN_DIR = "./fer2013/train"
TEST_DIR = "./fer2013/test"
IMG_SIZE = (48, 48)

# Define categories
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Preprocessing function
def preprocess_fer2013(directory, categories, img_size):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(directory, category)
        if not os.path.exists(path):
            print(f"Warning: Category folder '{category}' not found in {directory}")
            continue
        class_index = categories.index(category)
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            try:
                img = imread(img_path, as_gray=True)
                img_resized = resize(img, img_size, anti_aliasing=True)
                data.append(img_resized)
                labels.append(class_index)
            except Exception as e:
                print(f"Error processing file {img_file} in {path}: {e}")
    return np.array(data), np.array(labels)

# Load and preprocess the dataset
print("Preprocessing training data...")
X_train, y_train = preprocess_fer2013(TRAIN_DIR, CATEGORIES, IMG_SIZE)
print("Preprocessing test data...")
X_test, y_test = preprocess_fer2013(TEST_DIR, CATEGORIES, IMG_SIZE)

# Ensure dataset is not empty
if X_train.size == 0 or X_test.size == 0:
    raise ValueError("Training or test dataset is empty! Check the dataset paths and categories.")

# Reshape and normalize data
X_train = X_train.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1).astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, len(CATEGORIES))
y_test = to_categorical(y_test, len(CATEGORIES))

# Convert one-hot encoded labels to class indices for class weight calculation
y_train_indices = np.argmax(y_train, axis=1)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_indices), 
    y=y_train_indices
)

class_weight_dict = dict(zip(np.unique(y_train_indices), class_weights))
print("Class Weights:", class_weight_dict)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Build an improved model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(len(CATEGORIES), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Train the model
print("Training the model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save("C:/Users/kashv/Downloads/VIT Downloads/Project/FER2013_TRAINED_MODEL")

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=CATEGORIES))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Prediction for a single image
def predict_emotion(image_path, model, categories, img_size):
    img = imread(image_path, as_gray=True)
    img_resized = resize(img, img_size, anti_aliasing=True)
    img_array = img_resized.reshape(-1, img_size[0], img_size[1], 1).astype('float32') / 255.0
    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Example prediction
image_path = "C:/Users/kashv/Downloads/VIT Downloads/Project/image1.jpg"
predicted_class, confidence = predict_emotion(image_path, model, CATEGORIES, IMG_SIZE)
print(f"Predicted emotion: {predicted_class} with confidence {confidence:.2f}")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.show()

plot_history(history)
