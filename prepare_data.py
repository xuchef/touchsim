import touchsim as ts # import touchsim package
from touchsim.plotting import plot # import in-built plotting function
import numpy as np
import holoviews as hv # import holoviews package for plots and set some parameters
import os
import json
import pickle
import argparse
import shutil
from PIL import Image


def extract_category(string):
    return string.split("_", maxsplit=1)[1][:-4]


def main(args):
    idx = set()
    max_val = 0
    category_counts = {}
    for file in os.listdir(os.path.join(args.results_folder, "responses")):
        with open(os.path.join(args.results_folder, "responses", file), "rb") as f:
            r = pickle.load(f)

        if args.aff_class:
            r =  r[r.aff[args.aff_class]]

        idx |= set([i for i in range(len(r.spikes)) if len(r.spikes[i]) >= args.min_spikes_threshold])
        max_val = max(max_val, np.max(r.psth(args.bin_size)))

        category = extract_category(file)
        category_counts[category] = 1 if category not in category_counts else category_counts[category] + 1

    idx = np.array(list(idx))

    print("Num afferents:", len(idx))
    print("Max val:", max_val)

    training_folder_name = f"training_data_{args.aff_class}" if args.aff_class else "training_data"
    validation_folder_name = f"validation_data_{args.aff_class}" if args.aff_class else "validation_data"
    info_file_name = f"training_info_{args.aff_class}" if args.aff_class else "training_info" 

    training_folder_path = os.path.join(args.results_folder, training_folder_name)
    validation_folder_path = os.path.join(args.results_folder, validation_folder_name)

    shutil.rmtree(training_folder_path, ignore_errors=True)
    shutil.rmtree(validation_folder_path, ignore_errors=True)
    os.makedirs(training_folder_path, exist_ok=True)
    os.makedirs(validation_folder_path, exist_ok=True)

    category_counts_cur = {}
    for file in os.listdir(os.path.join(args.results_folder, "responses")):
        category = extract_category(file)
        category_counts_cur[category] = 1 if category not in category_counts_cur else category_counts_cur[category] + 1

        with open(os.path.join(args.results_folder, "responses", file), "rb") as f:
            r = pickle.load(f)

        if args.aff_class:
            r =  r[r.aff[args.aff_class]]

        os.makedirs(os.path.join(training_folder_path, category), exist_ok=True)
        os.makedirs(os.path.join(validation_folder_path, category), exist_ok=True)

        spike_histogram = r.psth(args.bin_size)
        spike_histogram = spike_histogram[idx]
        spike_histogram = np.interp(spike_histogram, (0, max_val), (0, 255))
        image = Image.fromarray(spike_histogram.astype(np.uint8), mode='L')
        directory = training_folder_name if (category_counts_cur[category]) / category_counts[category] * 100 <= args.percent_training else validation_folder_name
        image.save(os.path.join(args.results_folder, directory, category, f"{category_counts_cur[category]}.jpg"))

    print(category_counts)

    with open(os.path.join(args.results_folder, info_file_name), "w") as f:
        json.dump({
            "width": image.size[0],
            "height": image.size[1],
            "num_classes": len(category_counts)
        }, f)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Script for generating training data")

    # Add arguments to the parser
    parser.add_argument("--results_folder", type=str, help="Directory to obtain results from")
    parser.add_argument("--bin_size", type=int, help="The bin size (x-axis) in ms")
    parser.add_argument("--percent_training", type=float, help="The percent of data to allocate for training. Rest is validation.")
    parser.add_argument("--min_spikes_threshold", type=int, help="The minimum number of spikes to filter for")
    parser.add_argument("--aff_class", type=str, help="The type of afferent to use for learning")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args)
