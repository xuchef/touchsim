import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model

def main(args):
    model_weights_file_name = f"model_weights_{args.aff_class}_{args.batch_size}.h5" if args.aff_class else f"model_weights_{args.batch_size}"
    model = load_model(os.path.join(args.results_folder, model_weights_file_name))
    num_correct = 0
    total = 0

    for i, category in enumerate(os.listdir(os.path.join(args.results_folder, "validation_data"))):
        for file in os.listdir(os.path.join(args.results_folder, "validation_data", category)):
            test_image_path = os.path.join(args.results_folder, "validation_data", category, file)
            test_image = np.expand_dims(np.array(Image.open(test_image_path).convert("L")), axis=0)

            predictions = model.predict(test_image, verbose=0)
            predicted_class_index = np.argmax(predictions[0])

            print("X" if predicted_class_index else "O", end=" ")
            if predicted_class_index == i:
                num_correct += 1
            total += 1
        print()
    print(num_correct / total * 100)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Validate results")

    # Add arguments to the parser
    parser.add_argument("--results_folder", type=str, help="Directory to obtain results from")
    parser.add_argument("--aff_class", type=str, help="The type of afferent to use for learning")
    parser.add_argument("--batch_size", type=int, help="Training batch size")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args)