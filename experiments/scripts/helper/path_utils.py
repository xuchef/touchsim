import os
import datetime
from .constants import *

class PathUtil:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Shared timestamp for all instances
    
    def __init__(self, texture_set, dataset, aff_class=None):
        self.texture_set = texture_set
        self.dataset = dataset 
        self.aff_class = aff_class

        self.texture_set_dir = os.path.join(TEXTURE_SETS_DIR, self.texture_set)
        self.dataset_dir = os.path.join(DATASETS_DIR, self.dataset)

        self.responses_dir = os.path.join(self.dataset_dir, "responses")
        self.textures_dir = os.path.join(self.dataset_dir, "textures")
        self.stimuli_dir = os.path.join(self.dataset_dir, "stimuli")

        self.task_list_path = os.path.join(self.dataset_dir, "task_list")

        self.spikes_dir = os.path.join(self.dataset_dir, "spikes")
        self.aff_spikes_dirs = {aff_class : os.path.join(self.spikes_dir, aff_class) for aff_class in AFF_CHOICES}

        self.create_dataset_folders()

        self.aff_pop_path = os.path.join(self.dataset_dir, "aff_pop")

        if self.aff_class is not None:
            self.training_folder_path = os.path.join(self.dataset_dir, "training_data", self.aff_class)
            self.validation_folder_path = os.path.join(self.dataset_dir, "validation_data", self.aff_class)
            self.info_file_path = os.path.join(self.dataset_dir, "training_info", self.aff_class)

            self.model_weights_path = os.path.join(self.dataset_dir, "model_weights", self.aff_class, self.timestamp)
            self.log_dir = os.path.join(self.dataset_dir, "logs", "fit", self.aff_class, self.timestamp)

    def create_dataset_folders(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
        os.makedirs(self.textures_dir, exist_ok=True)
        os.makedirs(self.stimuli_dir, exist_ok=True)
        os.makedirs(self.spikes_dir, exist_ok=True)
        for dir_path in self.aff_spikes_dirs.values():
            os.makedirs(dir_path, exist_ok=True)


class NoSubdirectoriesError(Exception):
    def __init__(self, directory):
        self.directory = directory
        super().__init__(f"No subdirectories found in directory: {directory}")


def select_subdirectory(directory, prompt="Select a subdirectory"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    if not subdirectories:
        raise NoSubdirectoriesError(directory)

    print(prompt)
    print("-"*len(prompt))
    for idx, subdir in enumerate(subdirectories, start=1):
        subdir_path = os.path.join(directory, subdir)
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


if __name__ == "__main__":
    # p = PathUtil("hello", "hi")
    # p1 = PathUtil("HELLO", "HDLFHKL")

    # print(p.timestamp)
    # print(p1.timestamp)

    print(select_subdirectory(TEXTURE_SETS_DIR))
