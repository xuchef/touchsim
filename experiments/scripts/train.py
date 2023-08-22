import argparse
import os
from os.path import join
from PIL import Image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from .helper import *

def train_model(args, path_util, aff_class):
    train_path = path_util.aff_training_dirs[aff_class]
    test_path = path_util.aff_test_dirs[aff_class]

    image_sizes = load_json(path_util.image_sizes_path)
    img_width, img_height = image_sizes[aff_class]

    train_dataset, val_dataset = image_dataset_from_directory(
        train_path,
        label_mode="int", # experiment with non-sparse, "categorical"
        color_mode="grayscale",
        batch_size=args.batch_size,
        image_size=(img_height, img_width),
        validation_split=PERCENT_VALIDATION / (PERCENT_TRAINING + PERCENT_VALIDATION),
        subset="both",
        seed=SEED
    )

    test_dataset = image_dataset_from_directory(
        test_path,
        batch_size=args.batch_size,
        color_mode="grayscale",
        image_size=(img_height, img_width)
    )

    num_classes = len(train_dataset.class_names)

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

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(
        filepath=path_util.aff_weight_paths[aff_class],
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(
         log_dir=path_util.aff_logs_paths[aff_class],
         histogram_freq=1
    )

    # Train the model
    history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset,
                        callbacks=[checkpoint_callback, tensorboard_callback])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


def main(args):
    path_util = PathUtil().dataset(args.dataset).model(args.model)
    path_util.create_model_folders()

    for aff_class in AFF_CHOICES:
        print("\n", aff_class, "-"*30, sep="\n")
        train_model(args, path_util, aff_class)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN for texture classification based on neural spike trains jpg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str,
                        help="Subdirectory of datasets to train on")
    parser.add_argument("--model", type=str,
                        help="Subdirectory of models to store model in")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")

    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = select_subdirectory(DATASETS_DIR, "Select a dataset")
    
    if args.model is None:
            args.model = "CNN___" + args.dataset.split("___")[0]

    main(args)
