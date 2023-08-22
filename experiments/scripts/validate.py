import argparse
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from .helper import *


def plot_confusion_matrix(cm, class_names, aff_class, accuracy):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap="Blues")

    plt.title(f"{aff_class} Confusion Matrix", fontsize=16, pad=30)
    plt.text(0.5, 1.03, f"{accuracy*100:.2f}% accuracy", 
             fontsize=12, color="gray", ha="center", transform=plt.gca().transAxes)
    plt.colorbar(shrink=0.5)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=70, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    plt.xlabel("Predicted Class", fontsize=12, labelpad=20)
    plt.ylabel("True Class", fontsize=12, labelpad=20)

    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    return plt


def validate_model(path_util, aff_class):
    test_path = path_util.aff_test_dirs[aff_class]

    image_sizes = load_json(path_util.image_sizes_path)
    img_width, img_height = image_sizes[aff_class]

    test_dataset = image_dataset_from_directory(
        test_path,
        label_mode="int",
        batch_size=1,
        color_mode="grayscale",
        image_size=(img_height, img_width)
    )

    class_names = test_dataset.class_names
    class_names = [i.split("_")[0] for i in class_names]

    model = load_model(path_util.aff_weight_paths[aff_class])

    test_loss, test_accuracy = model.evaluate(test_dataset)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    test_images = []
    test_labels = []
    for images, labels in test_dataset:
        test_images.append(images)
        test_labels.append(labels)
    test_images = np.vstack(test_images)
    test_labels = np.vstack(test_labels).flatten()

    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    cm = confusion_matrix(test_labels, predicted_labels)
    report = classification_report(test_labels, predicted_labels, target_names=class_names)
    print(report)

    plot = plot_confusion_matrix(cm, class_names, aff_class, test_accuracy)
    plot.savefig(path_util.aff_confusion_matrix_paths[aff_class], dpi=300)
    plot.show()


def main(args):
    path_util = PathUtil().model(args.model)

    model_info = load_json(path_util.model_info_path)

    dataset = model_info["dataset"]
    path_util.dataset(dataset)

    for aff_class in AFF_CHOICES:
        print("\n", aff_class, "-"*30, sep="\n")
        validate_model(path_util, aff_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test CNN for texture classification based on neural spike trains jpg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=str,
                        help="Subdirectory of models to validate")

    args = parser.parse_args()

    if args.model is None:
        args.model = select_subdirectory(MODELS_DIR, "Select a model")

    main(args)
