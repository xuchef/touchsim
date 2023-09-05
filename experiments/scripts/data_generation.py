import os
from os.path import join
from  multiprocessing import Process, Value
import argparse
import numpy as np
import touchsim as ts
from .helper import *


def child_process(affpop, path_util, tasks, num_tasks_completed, num_total_tasks):
    """Process to run a subset of tasks.
    """
    for task in tasks:
        def file_path(parent_path):
            return join(parent_path, task["category"], str(task["id"]))

        texture = ts.Texture(filename=task["filename"], 
                             size=(task["x_upperbound"], task["y_upperbound"]), 
                             max_height=2)
        pins = ts.shape_hand(region="D2d_t", pins_per_mm=task["pins_per_mm"])
        stimulus = ts.stim_texture_indextip(texture, np.array(task["path_points"]), pins,
                                            depth=task["depth"], theta=task["theta"], 
                                            fs=task["sample_frequency"])
        response = affpop.response(stimulus)

        for attr in task["save_attrs"]:
            def save_filtered_response(func):
                for aff_class in AFF_CHOICES:
                    if aff_class in ts.constants.affclasses:
                        r = response[response.aff[aff_class]]
                    else:
                        r = response
                    path = getattr(path_util, path_util.dataset_aff_attr_names[attr])[aff_class]
                    save_pkl(func(r), file_path(path))

            if attr in SIMPLE_ATTRS:
                val = locals()[attr]
                path = file_path(getattr(path_util, path_util.dataset_attr_names[attr]))
                save_pkl(val, path)
            elif attr == "spikes":
                save_filtered_response(lambda r: r.spikes)
            elif attr == "psth":
                save_filtered_response(lambda r: r.psth(1))
            else:
                raise NotImplementedError(f"Custom attribute behaviour not implemented: {attr}")

        num_tasks_completed.value += 1
        print(f"{num_tasks_completed.value} / {num_total_tasks}", end="\r", flush=True)


def generate_path(x_upperbound, y_upperbound, distance, sample_count):
    """Let p1 be (0, 0) and p2 be [distance] away from p1 at a random angle on [0, PI/2].
    Then move p1 and p2 to a random position within the specified bounds.
    """
    axis_bounds = np.array([x_upperbound, y_upperbound])

    if (distance >= min(axis_bounds)):
        print("Distance must be less than axis bounds!")
        exit(1)
    
    p1 = np.zeros(2)
    theta = np.random.uniform(0, np.pi/2)
    p2 = np.array([np.cos(theta), np.sin(theta)]) * distance

    gap = axis_bounds - p2
    offset = np.random.uniform(0, gap)
    p1 += offset
    p2 += offset

    path_points = np.linspace(p1, p2, sample_count)

    return path_points


def main(args):
    # Initialize PathUtil object for standardizing path names across scripts
    path_util = PathUtil().dataset(args.dataset).dataset_folders(args.save_attrs).texture_set(args.texture_set)
    path_util.create_dataset_folders()

    # Obtain the texture names in the directory
    texture_names = os.listdir(path_util.texture_set_dir)

    # Extract the category names by removing the file extension
    categories = [remove_ext(i) for i in texture_names]
    path_util.create_category_folders(categories)

    # Save the parameters used to generate the dataset
    dataset_info = {**vars(args), "categories": categories}
    save_json(dataset_info, path_util.dataset_info_path)

    # Generate and save afferent population
    affpop = ts.affpop_hand(density_multiplier=args.affpop_density_multiplier)
    save_pkl(affpop, path_util.affpop_path)

    # Generate [samples_per_texture] paths and apply those paths along each texture
    task_list = []
    for i in range(args.samples_per_texture):
        path_points = generate_path(args.x_upperbound, args.y_upperbound, args.distance,
                                    int(args.sample_frequency * args.stimulus_duration))
        for file in texture_names:
            data_dict = {
                **vars(args),
                "filename": join(path_util.texture_set_dir, file),
                "category": remove_ext(file),
                "path_points": path_points,
                "id": i
            }
            task_list.append(data_dict)

    # Save the task list
    save_pkl(task_list, path_util.task_list_path)

    # Reduce the number of child processes if there aren't enough tasks to complete
    args.num_processes = min(len(task_list), args.num_processes)

    # Split tasks evenly among child processes, with the last process taking on the
    # remaining tasks if the number of tasks does not divide evenly
    items_per_child, remainder = divmod(len(task_list), args.num_processes)
    split_tasks = [task_list[i:i+items_per_child] 
                  for i in range(0, len(task_list), items_per_child)]
    if remainder > 0:
        split_tasks[-2].extend(split_tasks[-1])
        del split_tasks[-1]

    # Create a shared counter with atomic operations
    num_tasks_completed = Value("i", 0)

    # Create and run the child processes 
    processes = []
    num_total_tasks = len(task_list)
    for i in range(len(split_tasks)):
        process = Process(target=child_process,
                          args=(affpop, path_util, split_tasks[i], 
                                num_tasks_completed, num_total_tasks))
        processes.append(process)
        process.start()

    # Wait for all child processes to complete before continuing
    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate neural response data for a texture set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--texture_set", type=str,
                        help="Subdirectory of texture_sets to get textures from")
    parser.add_argument("--dataset", type=str,
                        help="Subdirectory of datasets to store results in")
    parser.add_argument("--save_attrs", nargs="+", choices=DATA_ATTR_CHOICES, default=CUSTOM_ATTRS,
                        help="Attributes to save")
    parser.add_argument("--num_processes", type=int, default=max(1, os.cpu_count()//4),
                        help="The number of processes to run")
    parser.add_argument("--stimulus_duration", type=float, default=1,
                        help="The duration for which each stimulus persists [seconds]")
    parser.add_argument("--sample_frequency", type=float, default=1000,
                        help="The stimulus sampling rate [Hz]")
    parser.add_argument("--samples_per_texture", type=int, default=1000,
                        help="The number of samples to be generated for each texture")
    parser.add_argument("--x_upperbound", type=int, default=60,
                        help="Width of texture")
    parser.add_argument("--y_upperbound", type=int, default=60,
                        help="Height of texture")
    parser.add_argument("--distance", type=float, default=50,
                        help="Required length of line")
    parser.add_argument("--pins_per_mm", type=float, default=1,
                        help="Stimulus pin density")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="Indentation depth [mm]")
    parser.add_argument("--theta", type=float, default=0,
                        help="Rotation of finger [rad]")
    parser.add_argument("--affpop_density_multiplier", type=float, default=1,
                        help="Factor to proportionally scale afferent density")

    args = parser.parse_args()

    if args.texture_set is None:
        args.texture_set = select_subdirectory(TEXTURE_SETS_DIR, "Select a texture set")

    if args.dataset is None:
        args.dataset = args.texture_set + "___" + PathUtil.timestamp 

    main(args)
