import touchsim as ts
import numpy as np
import os
from os.path import join
import argparse
from PIL import Image
from .helper import *


def get_threshold_indices(spikes, threshold):
    return set([i for i, s in enumerate(spikes) if len(s) >= threshold])


def binned_matrix(m, bin):
    remainder = m.shape[1] % bin
    padding = bin - remainder if remainder > 0 else 0
    m = np.pad(m, ((0,0), (0,padding)))
    return m.reshape(m.shape[0], -1, bin).sum(axis=2)


def get_attr_from_folder(attr, path_util, aff_class, category, index):
    path = join(getattr(path_util, path_util.dataset_aff_attr_names[attr])[aff_class],
                category,
                str(index))
    return load_pkl(path)


def save_image(image, path_util, aff_class, category, index, total):
    percent_done = index / total * 100
    if percent_done <= PERCENT_TRAINING + PERCENT_VALIDATION:
        directory = path_util.aff_training_dirs[aff_class]
    else:
        directory = path_util.aff_test_dirs[aff_class]
    image.save(os.path.join(directory, category, f"{index}.jpg"))


def run_manual_mode(args, path_util, dataset_info):
    raise NotImplementedError(f"Manual mode not implemented")


def run_quick_mode(args, path_util, dataset_info):
    image_sizes = {}

    for aff_class in AFF_CHOICES:
        valid_indices = set()
        max_val = 0

        print("\n", aff_class, "-"*30, sep="\n", flush=True)
        print("Pass #1 of 2", flush=True)
        for category in dataset_info["categories"]:
            for i in range(dataset_info["samples_per_texture"]):
                spikes = get_attr_from_folder("spikes", path_util, aff_class, category, i)
                psth = get_attr_from_folder("psth", path_util, aff_class, category, i)
                psth = binned_matrix(psth, args.bin_size)

                valid_indices |= get_threshold_indices(spikes, args.min_spikes_threshold)
                max_val = max(max_val, np.max(psth))

        valid_indices = np.array(list(valid_indices))
        print(f"{len(valid_indices)} indices with a max of {max_val}", flush=True)

        print("Pass #2 of 2", flush=True)
        for category in dataset_info["categories"]:
            for i in np.random.permutation(dataset_info["samples_per_texture"]):
                psth = get_attr_from_folder("psth", path_util, aff_class, category, i)
                psth = binned_matrix(psth, args.bin_size)

                psth = psth[valid_indices]
                psth = np.interp(psth, (0, max_val), (0, 255))
                image = Image.fromarray(psth.astype(np.uint8), mode="L")
                
                save_image(image, path_util, aff_class, category, i, dataset_info["samples_per_texture"])

        image_sizes[aff_class] = image.size

    save_json(image_sizes, path_util.image_sizes_path)

    #     dir_path = path_util.dataset_attr_names["response"]
    #     response_paths = [join(dir_path, i) for i in os.listdir(dir_path)]
    #     np.random.shuffle(response_paths)

        

        
    #     for i, path in enumerate(response_paths):
    #         print(f"{i}/{len(response_paths)}", end="\r", flush=True)

    #         r = load_pkl(path)
    #         r = filter_aff_class(r, aff_class)
    #         valid_indices |= get_threshold_indices(r.spikes, args.min_spikes_threshold)
    #         max_val = max(max_val, np.max(r.psth(args.bin_size)))

    #     valid_indices = np.array(list(valid_indices))
    #     print(f"{len(valid_indices)} indices with a max of {max_val}")


    #     print("Pass #2 of 2")
    #     category_counts_cur = defaultdict(int)
    #     for i, path in enumerate(response_paths):
    #         print(f"{i}/{len(response_paths)}", end="\r", flush=True)
    #         category = extract_category(path)
    #         category_counts_cur[category] += 1
    #         i = category_counts_cur[category] 

    #         r = load_pkl(path)
    #         r = filter_aff_class(r, aff_class)

    #         spike_histogram = r.psth(args.bin_size)
    #         spike_histogram = spike_histogram[valid_indices]
    #         spike_histogram = np.interp(spike_histogram, (0, max_val), (0, 255))
    #         image = Image.fromarray(spike_histogram.astype(np.uint8), mode='L')

    #         percent_done = i / category_counts[category] * 100
    #         if percent_done <= PERCENT_TRAINING + PERCENT_VALIDATION:
    #             directory = path_util.aff_training_dirs[aff_class]
    #         else:
    #             directory = path_util.aff_test_dirs[aff_class]
    #         image.save(os.path.join(directory, category, f"{i}.jpg"))
    #     image_sizes[aff_class] = image.size
    #     print("-"*30)
    # save_json(image_sizes, path_util.image_sizes_path)


def main(args):
    # Initialize PathUtil object for standardizing path names across scripts
    path_util = PathUtil().dataset(args.dataset)
    dataset_info = load_json(path_util.dataset_info_path)

    quick_mode = {"spikes", "psth"}.issubset(dataset_info["save_attrs"])
    manual_mode = "response" in dataset_info["save_attrs"]

    assert quick_mode or manual_mode

    path_util.dataset_folders(dataset_info["save_attrs"] + ["training", "test"])
    path_util.create_category_folders(dataset_info["categories"])

    if quick_mode:
        run_quick_mode(args, path_util, dataset_info)
    else:
        run_manual_mode(args, path_util, dataset_info)


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
