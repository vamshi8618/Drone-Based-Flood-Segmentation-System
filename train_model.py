import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the U-Net model
def unet_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(b)

    # Decoder
    u2 = layers.UpSampling2D((2, 2))(b)
    u2 = layers.Conv2D(128, (2, 2), activation='relu', padding='same')(u2)
    u2 = layers.Concatenate()([u2, c2])

    u1 = layers.UpSampling2D((2, 2))(u2)
    u1 = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(u1)
    u1 = layers.Concatenate()([u1, c1])

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u1)

    model = models.Model(inputs, outputs)
    return model


# Load and preprocess the dataset
def load_data(image_dir, mask_dir, image_size=(256, 256)):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    for img_file, mask_file in zip(image_files, mask_files):
        # Load and preprocess the image
        img = load_img(os.path.join(image_dir, img_file), target_size=image_size)
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)
        
        # Load and preprocess the mask
        mask = load_img(os.path.join(mask_dir, mask_file), target_size=image_size, color_mode='grayscale')
        mask = img_to_array(mask) / 255.0  # Normalize to [0, 1]
        masks.append(mask)
    
    return np.array(images), np.array(masks)


# Main script
if __name__ == '__main__':
    # Define dataset paths
    train_image_dir = 'dataset/train/images'
    train_mask_dir = 'dataset/train/masks'

    # Load training data
    X_train, y_train = load_data(train_image_dir, train_mask_dir)

    # Ensure masks have the correct shape (expand dimensions if needed)
    y_train = np.expand_dims(y_train, axis=-1)

    # Create the model
    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model without validation data
    model.fit(X_train, y_train, epochs=10, batch_size=16)

    # Save the model
    model.save('unet_model.h5')
