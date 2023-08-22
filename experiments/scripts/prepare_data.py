import touchsim as ts
import numpy as np
import os
from os.path import join
import argparse
from PIL import Image
from .helper import *
import re
from collections import defaultdict


def extract_category(path):
    return re.search(r"/\d+_(.+?)(?:\.\w+)+$", path).group(1)


def filter_aff_class(r, aff_class):
    if aff_class not in ts.constants.affclasses:
        return r
    return r[r.aff[aff_class]]


def get_threshold_indices(spikes, threshold):
    return set([i for i, s in enumerate(spikes) if len(s) >= threshold])


def main(args):
    # Initialize PathUtil object for standardizing path names across scripts
    path_util = PathUtil().dataset(args.dataset)

    image_sizes = {}

    for aff_class in AFF_CHOICES:
        print(aff_class)
        dir_path = path_util.responses_dir
        response_paths = [join(dir_path, i) for i in os.listdir(dir_path)]
        np.random.shuffle(response_paths)

        valid_indices = set()
        max_val = 0
        category_counts = defaultdict(int)

        print("Pass #1 of 2")
        for i, path in enumerate(response_paths):
            print(f"{i}/{len(response_paths)}", end="\r", flush=True)
            category = extract_category(path)
            category_counts[category] += 1

            r = load_pkl(path)
            r = filter_aff_class(r, aff_class)
            valid_indices |= get_threshold_indices(r.spikes, args.min_spikes_threshold)
            max_val = max(max_val, np.max(r.psth(args.bin_size)))

        valid_indices = np.array(list(valid_indices))
        print(f"{len(valid_indices)} indices with a max of {max_val}")

        path_util.create_category_folders(category_counts.keys())

        print("Pass #2 of 2")
        category_counts_cur = defaultdict(int)
        for i, path in enumerate(response_paths):
            print(f"{i}/{len(response_paths)}", end="\r", flush=True)
            category = extract_category(path)
            category_counts_cur[category] += 1
            i = category_counts_cur[category] 

            r = load_pkl(path)
            r = filter_aff_class(r, aff_class)

            spike_histogram = r.psth(args.bin_size)
            spike_histogram = spike_histogram[valid_indices]
            spike_histogram = np.interp(spike_histogram, (0, max_val), (0, 255))
            image = Image.fromarray(spike_histogram.astype(np.uint8), mode='L')

            percent_done = i / category_counts[category] * 100
            if percent_done <= PERCENT_TRAINING + PERCENT_VALIDATION:
                directory = path_util.aff_training_dirs[aff_class]
            else:
                directory = path_util.aff_test_dirs[aff_class]
            image.save(os.path.join(directory, category, f"{i}.jpg"))
        image_sizes[aff_class] = image.size
        print("-"*30)
    save_json(image_sizes, path_util.image_sizes_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert neural spike trains into jpg images to be used for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str,
                        help="Subdirectory of datasets to get results from")
    parser.add_argument("--bin_size", type=int, default=5,
                        help="The bin size for discretizing time [ms]")
    parser.add_argument("--min_spikes_threshold", type=int, default=5,
                        help="Filter out afferents with less total spikes than MIN_SPIKES_THRESHOLD")

    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = select_subdirectory(DATASETS_DIR, "Select a dataset")

    main(args)
