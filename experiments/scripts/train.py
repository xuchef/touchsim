import numpy as np
import os
import json
import tensorflow as tf
import argparse
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime


def main(args):
    training_folder_path = os.path.join(args.results_folder, "training_data", args.aff_class)
    validation_folder_path = os.path.join(args.results_folder, "validation_data", args.aff_class)
    info_file_path = os.path.join(args.results_folder, "training_info", args.aff_class)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model_weights_path = os.path.join(args.results_folder, "model_weights", args.aff_class, timestamp)

    log_dir = os.path.join(args.results_folder, "logs", "fit", args.aff_class, timestamp)

    with open(info_file_path) as f:
        training_info = json.load(f)

    # Define image dimensions and categories
    img_width = training_info["width"]
    img_height = training_info["height"]
    num_classes = training_info["num_classes"]

    # Create data generators with data augmentation for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        training_folder_path,
        target_size=(img_height, img_width),
        batch_size=args.batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_folder_path,
        target_size=(img_height, img_width),
        batch_size=args.batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    # Create a CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # model = Sequential([
    #     Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(256, (3, 3), activation='relu'),  # Additional layer
    #     MaxPooling2D((2, 2)),
    #     Conv2D(512, (3, 3), activation='relu'),  # Additional layer
    #     MaxPooling2D((2, 2)),
    #     Flatten(),
    #     Dense(512, activation='relu'),  # Increased units
    #     Dense(128, activation='relu'),
    #     Dense(num_classes, activation='softmax')
    # ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(
        filepath=model_weights_path,
        save_best_only=True,  # Only save the model with the best validation accuracy
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',  # Mode can be 'min', 'max', or 'auto'
        verbose=1  # Display messages when saving models
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = model.fit(train_generator, epochs=args.epochs, validation_data=validation_generator,
                        callbacks=[checkpoint_callback, tensorboard_callback])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(validation_generator)
    print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Train CNN for binary texture classification based on neuron firing image")

    # Add arguments to the parser
    parser.add_argument("--results_folder", type=str, help="Directory to obtain results from", required=True)
    parser.add_argument("--aff_class", type=str, choices=['RA', 'SA1', 'PC', 'all'], help="The type of afferent to use for learning", required=True)
    parser.add_argument("--epochs", type=int, help="Number of training epochs", required=True)
    parser.add_argument("--batch_size", type=int, help="Training batch size", required=True)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args)
