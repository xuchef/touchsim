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

# args.results_folder = "results_classification_1ppm_3"
# args.bin_size = 5
# args.percent_training = 80
# args.min_spikes_threshold = 1


def extract_category(string):
    return string.split("_", maxsplit=1)[1][:-4]


def main(args):
    if os.path.exists(os.path.join(args.results_folder, "response_dict.pkl")):
        with open(os.path.join(args.results_folder, "response_dict.pkl"), "rb") as f:
            response_dict = pickle.load(f)
    else:
        response_dict = {}
        for file in os.listdir(os.path.join(args.results_folder, "responses")):
            with open(os.path.join(args.results_folder, "responses", file), "rb") as f:
                r = pickle.load(f)
            response_dict[file] = r
        with open(os.path.join(args.results_folder, "response_dict.pkl"), "wb") as f:
            pickle.dump(response_dict,f)

    if args.aff_class:
        response_dict = {key: r[r.aff[args.aff_class]] for key, r in response_dict.items()}

    idx = set()
    for r in response_dict.values():
        idx |= set([i for i in range(len(r.spikes)) if len(r.spikes[i]) >= args.min_spikes_threshold])
    idx = np.array(list(idx))
    print("Num afferents:", len(idx))

    # lens = [np.count_nonzero([arr.size > args.min_spikes_threshold for arr in r.spikes]) for r in response_dict.values()]
    # np.median(lens)

    max_val = 0
    for r in response_dict.values():
        max_val = max(max_val, np.max(r.psth(args.bin_size)))
    print("Max val:", max_val)


    categorized_responses = {}
    for key, value in response_dict.items():
        category = extract_category(key)
        if category not in categorized_responses:
            categorized_responses[category] = []
        categorized_responses[category].append(value)

    for val in categorized_responses.values():
        np.random.shuffle(val)

    training_folder_name = f"training_data_{args.aff_class}" if args.aff_class else "training_data"
    validation_folder_name = f"validation_data_{args.aff_class}" if args.aff_class else "validation_data"
    info_file_name = f"training_info_{args.aff_class}" if args.aff_class else "training_info" 

    training_folder_path = os.path.join(args.results_folder, training_folder_name)
    validation_folder_path = os.path.join(args.results_folder, validation_folder_name)

    shutil.rmtree(training_folder_path, ignore_errors=True)
    shutil.rmtree(validation_folder_path, ignore_errors=True)
    os.makedirs(training_folder_path, exist_ok=True)
    os.makedirs(validation_folder_path, exist_ok=True)

    for category, responses in categorized_responses.items():
        os.makedirs(os.path.join(training_folder_path, category), exist_ok=True)
        os.makedirs(os.path.join(validation_folder_path, category), exist_ok=True)

        for i, r in enumerate(responses):
            spike_histogram = r.psth(args.bin_size)
            spike_histogram = spike_histogram[idx]
            spike_histogram = np.interp(spike_histogram, (0, max_val), (0, 255))
            image = Image.fromarray(spike_histogram.astype(np.uint8), mode='L')#.resize((200, 200))
            directory = training_folder_name if (i+1) / len(responses) * 100 <= args.percent_training else validation_folder_name
            image.save(os.path.join(args.results_folder, directory, category, f"{i}.jpg"))

    with open(os.path.join(args.results_folder, info_file_name), "w") as f:
        json.dump({
            "width": image.size[0],
            "height": image.size[1],
            "num_classes": len(categorized_responses)
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
