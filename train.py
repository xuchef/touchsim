import numpy as np
import os
import json
import tensorflow as tf
import argparse
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

def main(args):
    training_folder_name = f"training_data_{args.aff_class}" if args.aff_class else "training_data"
    validation_folder_name = f"validation_data_{args.aff_class}" if args.aff_class else "validation_data"
    info_file_name = f"training_info_{args.aff_class}" if args.aff_class else "training_info" 
    model_weights_file_name = f"model_weights_{args.aff_class}.h5" if args.aff_class else "model_weights"

    with open(os.path.join(args.results_folder, info_file_name)) as f:
        training_info = json.load(f)

    # Define image dimensions and categories
    img_width = training_info["width"]
    img_height = training_info["height"]
    num_classes = training_info["num_classes"]

    # Load your dataset and split it into training and validation sets
    train_data_dir = os.path.join(args.results_folder, training_folder_name)
    validation_data_dir = os.path.join(args.results_folder, validation_folder_name)

    # Create data generators with data augmentation for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=args.batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
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
        filepath=os.path.join(args.results_folder, model_weights_file_name),
        save_best_only=True,  # Only save the model with the best validation accuracy
        monitor='val_accuracy',  # Monitor validation accuracy
        mode='max',  # Mode can be 'min', 'max', or 'auto'
        verbose=1  # Display messages when saving models
    )

    # Train the model
    history = model.fit(train_generator, epochs=args.epochs, validation_data=validation_generator, callbacks=[checkpoint_callback])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(validation_generator)
    print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Train CNN for binary texture classification based on neuron firing image")

    # Add arguments to the parser
    parser.add_argument("--results_folder", type=str, help="Directory to obtain results from")
    parser.add_argument("--aff_class", type=str, help="The type of afferent to use for learning")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args)
