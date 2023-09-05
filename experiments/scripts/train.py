import argparse
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from .helper import *
from .validate import validate_model


def load_train_val_data(path, image_size, batch_size):
    return image_dataset_from_directory(
        path,
        label_mode="int", # experiment with non-sparse, "categorical"
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size,
        validation_split=PERCENT_VALIDATION / (PERCENT_TRAINING + PERCENT_VALIDATION),
        subset="both",
        seed=SEED
    )


def load_test_data(path, image_size):
    return image_dataset_from_directory(
        path,
        batch_size=1,
        color_mode="grayscale",
        image_size=image_size
    )


def create_CNN_model(num_classes, image_size):
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])


def callback_model_checkpoint(path):
    return ModelCheckpoint(
        filepath=path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )


def callback_tensorboard(path):
    return TensorBoard(
         log_dir=path,
         histogram_freq=1
    )


def train_model(args, path_util, aff_class):
    img_width, img_height = load_json(path_util.image_sizes_path)[aff_class]
    image_size = (img_height, img_width)

    train_dataset, val_dataset = load_train_val_data(path_util.aff_training_dirs[aff_class], image_size, args.batch_size)
    test_dataset = load_test_data(path_util.aff_training_dirs[aff_class], image_size)

    model = create_CNN_model(len(train_dataset.class_names), image_size)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset,
              callbacks=[
                  callback_model_checkpoint(path_util.aff_weight_paths[aff_class]),
                  callback_tensorboard(path_util.aff_logs_paths[aff_class])
              ])

    validate_model(path_util, aff_class)


def main(args):
    path_util = PathUtil().dataset(args.dataset).model(args.model)
    path_util.create_model_folders()

    model_info = {"dataset": args.dataset}
    save_json(model_info, path_util.model_info_path)

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
