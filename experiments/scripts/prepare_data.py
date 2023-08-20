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
    training_folder_path = os.path.join(args.results_folder, "training_data", args.aff_class)
    validation_folder_path = os.path.join(args.results_folder, "validation_data", args.aff_class)
    info_file_path = os.path.join(args.results_folder, "training_info", args.aff_class)

    responses_folder_path = os.path.join(args.results_folder, "responses")

    idx = set()
    max_val = 0
    category_counts = {}
    print("Get population data")
    responses = os.listdir(responses_folder_path)
    for i, file in enumerate(responses):
        print(i, "/", len(responses), end="\r")
        with open(os.path.join(responses_folder_path, file), "rb") as f:
            r = pickle.load(f)

        if args.aff_class in ts.constants.affclasses:
            r =  r[r.aff[args.aff_class]]

        idx |= set([i for i in range(len(r.spikes)) if len(r.spikes[i]) >= args.min_spikes_threshold])
        max_val = max(max_val, np.max(r.psth(args.bin_size)))

        category = extract_category(file)
        category_counts[category] = 1 if category not in category_counts else category_counts[category] + 1

    idx = np.array(list(idx))

    print("Num afferents:", len(idx))
    print("Max val:", max_val)

    shutil.rmtree(training_folder_path, ignore_errors=True)
    shutil.rmtree(validation_folder_path, ignore_errors=True)
    os.makedirs(training_folder_path, exist_ok=True)
    os.makedirs(validation_folder_path, exist_ok=True)

    print("Generate images")
    category_counts_cur = {}
    for i, file in enumerate(responses):
        print(i, "/", len(responses), end="\r")
        category = extract_category(file)
        category_counts_cur[category] = 1 if category not in category_counts_cur else category_counts_cur[category] + 1

        with open(os.path.join(responses_folder_path, file), "rb") as f:
            r = pickle.load(f)

        if args.aff_class in ts.constants.affclasses:
            r =  r[r.aff[args.aff_class]]

        os.makedirs(os.path.join(training_folder_path, category), exist_ok=True)
        os.makedirs(os.path.join(validation_folder_path, category), exist_ok=True)

        spike_histogram = r.psth(args.bin_size)
        spike_histogram = spike_histogram[idx]
        spike_histogram = np.interp(spike_histogram, (0, max_val), (0, 255))
        image = Image.fromarray(spike_histogram.astype(np.uint8), mode='L')
        directory = training_folder_path if (category_counts_cur[category]) / category_counts[category] * 100 <= args.percent_training else validation_folder_path
        image.save(os.path.join(directory, category, f"{category_counts_cur[category]}.jpg"))

    print(category_counts)

    with open(info_file_path, "w") as f:
        json.dump({
            "width": image.size[0],
            "height": image.size[1],
            "num_classes": len(category_counts)
        }, f)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Script for generating training data")

    # Add arguments to the parser
    parser.add_argument("--results_folder", type=str, help="Directory to obtain results from", required=True)
    parser.add_argument("--bin_size", type=int, help="The bin size (x-axis) in ms", required=True)
    parser.add_argument("--percent_training", type=float, help="The percent of data to allocate for training. Rest is validation.", required=True)
    parser.add_argument("--min_spikes_threshold", type=int, help="The minimum number of spikes to filter for", required=True)
    parser.add_argument("--aff_class", type=str, choices=['RA', 'SA1', 'PC', 'all'], help="The type of afferent to use for learning", required=True)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args)
