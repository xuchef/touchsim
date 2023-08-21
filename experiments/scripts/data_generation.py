import os
from  multiprocessing import Process
import argparse
import numpy as np
import touchsim as ts
from .helper import *


def child_process(affpop, path_util, tasks, num_total_tasks):
    """Process to run a subset of tasks. Note that all arguments should be read-only.
    """
    for task in tasks:
        texture = ts.Texture(filename=task["filename"], 
                             size=(task["x_upperbound"], task["y_upperbound"]), 
                             max_height=2)
        pins = ts.shape_hand(region="D2d_t", pins_per_mm=task["pins_per_mm"])
        stimulus = ts.stim_texture_indextip(texture, np.array(task["path_points"]), pins,
                                            depth=task["depth"], theta=task["theta"], 
                                            fs=task["sample_frequency"])
        response = affpop.response(stimulus)

        save_pkl(response, os.path.join(path_util.responses_dir, task["id"]))
        save_pkl(texture, os.path.join(path_util.textures_dir, task["id"]))
        save_pkl(stimulus, os.path.join(path_util.stimuli_dir, task["id"]))

        for aff_class, dir_path in path_util.aff_spikes_dirs.items():
            if aff_class in ts.constants.affclasses:
                spikes = response[response.aff[aff_class]].spikes
            else:
                spikes = response.spikes
            save_pkl(spikes, os.path.join(dir_path, task["id"]))

        num_tasks_completed = len(os.listdir(path_util.responses_dir))
        print(f"{num_tasks_completed} / {num_total_tasks}")


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
    path_util = PathUtil(args.texture_set, args.dataset)

    # Generate and save afferent population
    affpop = ts.affpop_hand(density_multiplier=args.affpop_density_multiplier)
    save_pkl(affpop, path_util.aff_pop_path)

    # Generate [samples_per_texture] paths and apply those paths along each texture
    task_list = []
    texture_names = os.listdir(path_util.texture_set_dir)
    for i in range(args.samples_per_texture):
        path_points = generate_path(args.x_upperbound, args.y_upperbound, args.distance,
                                    int(args.sample_frequency * args.stimulus_duration))
        for file in texture_names:
            data_dict = {
                **vars(args),
                "filename": os.path.join(path_util.texture_set_dir, file),
                "path_points": path_points,
                "id": f"{i}_{file}" 
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

    # Create and run the child processes 
    processes = []
    num_total_tasks = len(task_list)
    for i in range(len(split_tasks)):
        process = Process(target=child_process,
                          args=(affpop, path_util, split_tasks[i], num_total_tasks))
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
                        help="Subdirectory of 'datasets' to store results in")
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
