import os
from os.path import join
import datetime
from .constants import *
from .data_utils import *

class PathUtil:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Shared timestamp for all instances

    def texture_set(self, texture_set):
        self.texture_set_dir = join(TEXTURE_SETS_DIR, texture_set)
        return self

    def dataset(self, dataset):
        self.dataset_dir = join(DATASETS_DIR, dataset)

        self.aff_pop_path = join(self.dataset_dir, "aff_pop")
        self.task_list_path = join(self.dataset_dir, "task_list")
        self.image_sizes_path = join(self.dataset_dir, "image_sizes")
        self.responses_dir = join(self.dataset_dir, "responses")
        self.textures_dir = join(self.dataset_dir, "textures")
        self.stimuli_dir = join(self.dataset_dir, "stimuli")

        self.spikes_dir = join(self.dataset_dir, "spikes")
        self.aff_spikes_dirs = aff_dirs(self.spikes_dir)

        self.psth_dir = join(self.dataset_dir, "psth")
        self.aff_psth_dirs = aff_dirs(self.psth_dir)
    
        self.training_dir = join(self.dataset_dir, "training")
        self.aff_training_dirs = aff_dirs(self.training_dir)

        self.test_dir = join(self.dataset_dir, "test")
        self.aff_test_dirs = aff_dirs(self.test_dir)

        return self

    def model(self, model):
        self.model_dir = join(MODELS_DIR, model)

        self.model_info_path = join(self.model_dir, "model_info")

        self.aff_model_dirs = aff_dirs(self.model_dir)

        self.aff_weight_paths = aff_dirs_dict(self.aff_model_dirs, "weights.keras")
        self.aff_logs_paths = aff_dirs_dict(self.aff_model_dirs, "logs")
        self.aff_confusion_matrix_paths = aff_dirs_dict(self.aff_model_dirs, "confusion_matrix.png")

        return self

    def create_dataset_folders(self, save_all=True):
        assert self.dataset_dir is not None
        paths = dict_vals_to_list(
            self.aff_spikes_dirs,
            self.aff_psth_dirs,
            self.aff_training_dirs,
            self.aff_test_dirs
        )
        paths.append(self.responses_dir)
        if save_all:
            paths += [self.textures_dir, self.stimuli_dir]
        make_all_dirs(paths)
        return self

    def create_model_folders(self):
        assert self.model_dir is not None
        paths = dict_vals_to_list(
            self.aff_model_dirs,
            self.aff_logs_paths
        )
        make_all_dirs(paths)
        return self

    def create_category_folders(self, categories):
        assert self.dataset_dir is not None
        paths = dict_vals_to_list(
            *[aff_dirs_dict(self.aff_training_dirs, c) for c in categories],
            *[aff_dirs_dict(self.aff_test_dirs, c) for c in categories]
        )
        make_all_dirs(paths)


def dict_vals_to_list(*args):
    paths = []
    for d in args:
        paths += list(d.values())
    return paths


def make_all_dirs(dirs):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def aff_dirs_func(func):
    return {aff_class : join(*func(aff_class)) for aff_class in AFF_CHOICES}


def aff_dirs(path):
    return aff_dirs_func(lambda a: [path, a])


def aff_dirs_dict(dict, path):
    return aff_dirs_func(lambda a: [dict[a], path])


class NoSubdirectoriesError(Exception):
    def __init__(self, directory):
        self.directory = directory
        super().__init__(f"No subdirectories found in directory: {directory}")


def select_subdirectory(directory, prompt="Select a subdirectory"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(join(directory, d))]
    
    if not subdirectories:
        raise NoSubdirectoriesError(directory)

    print(prompt)
    print("-"*len(prompt))
    for idx, subdir in enumerate(subdirectories, start=1):
        subdir_path = join(directory, subdir)
        num_items = len(os.listdir(subdir_path))
        print(f"{idx}) {subdir} ({num_items} items)")
    print()
    
    while True:
        try:
            choice = int(input("Option number: "))
            if 1 <= choice <= len(subdirectories):
                return subdirectories[choice - 1]
            else:
                print("Invalid option. Please choose a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
