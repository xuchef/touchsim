import os
from os.path import join
import datetime
from .constants import *
from .data_utils import *


class PathUtil:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Shared timestamp for all instances

    def __init__(self):
        self.dataset_attr_names = {}
        self.dataset_aff_attr_names = {}

    def getattrs(self, attrs):
        return [getattr(self, a) for a in attrs]

    def texture_set(self, texture_set):
        self.texture_set_dir = join(TEXTURE_SETS_DIR, texture_set)
        return self

    def add_dataset_dir(self, name):
        assert self.dataset_dir is not None

        attr_name = f"{name}_dir"
        path = join(self.dataset_dir, name)
        setattr(self, attr_name, path)
        self.dataset_attr_names[name] = attr_name
        return path
    
    def add_dataset_dir_with_affs(self, name):
        assert self.dataset_dir is not None

        attr_dir_path = self.add_dataset_dir(name)
        self.dataset_attr_names.pop(name)
        
        aff_attr_name = f"aff_{name}_dirs"
        aff_paths = aff_dirs(attr_dir_path)
        setattr(self, aff_attr_name, aff_paths)
        self.dataset_aff_attr_names[name] = aff_attr_name

    def dataset(self, dataset):
        self.dataset_dir = join(DATASETS_DIR, dataset)

        self.dataset_info_path = join(self.dataset_dir, "dataset_info")
        self.affpop_path = join(self.dataset_dir, "affpop")
        self.task_list_path = join(self.dataset_dir, "task_list")
        self.image_sizes_path = join(self.dataset_dir, "image_sizes")

        return self

    def dataset_folders(self, folder_names):
        assert self.dataset_dir is not None

        for name in folder_names:
            if name in SIMPLE_ATTRS:
                self.add_dataset_dir(name)
            elif name in ["spikes", "psth", "training", "test"]:
                self.add_dataset_dir_with_affs(name)
            else:
                raise NotImplementedError(f"Custom folder structure not implemented: {name}")

        return self

    def model(self, model):
        self.model_dir = join(MODELS_DIR, model)

        self.aff_model_dirs = aff_dirs(self.model_dir)

        self.model_info_path = join(self.model_dir, "model_info")

        self.aff_logs_dirs = aff_dirs_dict(self.aff_model_dirs, "logs")
        self.aff_weight_paths = aff_dirs_dict(self.aff_model_dirs, "weights.keras")
        self.aff_confusion_matrix_paths = aff_dirs_dict(self.aff_model_dirs, "confusion_matrix.png")

        return self

    def create_dataset_folders(self):
        assert self.dataset_dir is not None
        paths = self.getattrs(self.dataset_attr_names.values())
        paths += dict_vals_to_list(*self.getattrs(self.dataset_aff_attr_names.values()))
        make_all_dirs(paths)
        return self

    def create_model_folders(self):
        assert self.model_dir is not None
        paths = dict_vals_to_list(
            self.aff_model_dirs,
            self.aff_logs_dirs
        )
        make_all_dirs(paths)
        return self

    def create_category_folders(self, categories):
        assert self.dataset_dir is not None
        for dir in self.getattrs(self.dataset_attr_names.values()):
            paths = [join(dir, c) for c in categories]
            make_all_dirs(paths)
        for dir in self.getattrs(self.dataset_aff_attr_names.values()):
            paths = dict_vals_to_list(*[aff_dirs_dict(dir, c) for c in categories])
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
