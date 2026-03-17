import tensorflow as tf
from tensorflow.keras import layers, models
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 32
img_height = 32
img_width = 32
data_dir = "dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training",
    seed=123, color_mode="grayscale", image_size=(img_height, img_width), batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation",
    seed=123, color_mode="grayscale", image_size=(img_height, img_width), batch_size=batch_size)

num_classes = len(train_ds.class_names)

# NEW: Data Augmentation Layer (Makes the AI robust to different image styles)
data_augmentation = tf.keras.Sequential([
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
])

model = models.Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
    data_augmentation,  # Apply scrambler here

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),  # Prevents memorization
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=15)
model.save('braille_cnn.keras')
print("Robust Model trained and saved!")