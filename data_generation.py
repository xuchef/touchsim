import os
import multiprocessing
import argparse
import numpy as np
import pickle
import touchsim as ts
import json
from touchsim.plotting import plot


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def child_process(a, results_folder, process_name, tasks):
    for task in tasks:
        texture = ts.Texture(filename=task['file_path'], size=task['bounds'], max_height=1)  # keep max_height at 1
        
        # plot(texture, locs=np.array(task['path_points']))
        pins = ts.shape_hand(region="D2d_t", pins_per_mm=task['pins_per_mm'])
        s = ts.stim_texture_indextip(texture, np.array(task['path_points']), pins, 
            depth=task['depth'], theta=task['theta'], fs=task['sample_frequency'])

        r = a.response(s)

        save_obj(r, os.path.join(results_folder, 'responses', task['id']+'.pkl'))
        save_obj(texture, os.path.join(results_folder, 'textures', task['id']+'.pkl'))
        save_obj(s, os.path.join(results_folder, 'stimuli', task['id']+'.pkl'))

        print(task['file_path'], r, sep="\n")

def within_bounds(point, axis_bounds):
    x_within_bounds = (point[0] >= 0 and point[0] <= axis_bounds[0])
    y_within_bounds = (point[1] >= 0 and point[1] <= axis_bounds[1])
    return (x_within_bounds and y_within_bounds)

def generate_path(x_upperbound, y_upperbound, distance, sample_count):
    axis_bounds = (x_upperbound, y_upperbound)
    if (distance >= x_upperbound or distance >= y_upperbound):
        print('Distance must be within axis bounds!')
        exit(1)
    
    p1 = (np.random.uniform(0, axis_bounds[0]), np.random.uniform(0, axis_bounds[1]))
    theta = np.random.uniform(0, 2 * np.pi)
    p2 = (p1[0] + distance * np.cos(theta), p1[1] + distance * np.sin(theta))
    if p2[0] < 0 or p2[0] >= axis_bounds[0] or p2[1] < 0 or p2[1] >= axis_bounds[1]:
        # Shift p2 to be within axis_bounds
        buffer = np.random.uniform(0, min(axis_bounds) - distance)
        p2 = (min(max(p2[0], 0), axis_bounds[0] - buffer), min(max(p2[1], 0), axis_bounds[1] - buffer))
        p1 = (p2[0] - distance * np.cos(theta), p2[1] - distance * np.sin(theta))
    # print(p1, p2, np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
    # lastly, to sample sample_count points on this line
    x_samples = np.linspace(p1[0], p2[0], sample_count)
    y_samples = np.linspace(p1[1], p2[1], sample_count)
    points = []
    for i in range(sample_count):
        points.append((x_samples[i], y_samples[i]))
    return points

def get_afferent_pop(results_folder):
    aff_pop_file_path = os.path.join(results_folder, 'afferent_pop.pkl')
    if os.path.exists(aff_pop_file_path):
        with open(aff_pop_file_path, "rb") as f:
            a = pickle.load(f)
    else:
        a = ts.affpop_hand(density_multiplier=1)
        with open(aff_pop_file_path, "wb") as f:
            pickle.dump(a, f)
    return a



def main(args):
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, 'responses'), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, 'textures'), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, 'stimuli'), exist_ok=True)
    a = get_afferent_pop(args.results_folder)
    path_list = []
    contents = os.listdir(args.texture_file_directory)
    contents = contents[:args.texture_count]
    for i in range(args.samples_per_texture):
        path_points = generate_path(args.x_upperbound, args.y_upperbound, args.distance, 
                                    int(args.sample_frequency * args.stimulus_duration))
        for file in contents:
            data_dict = {}
            file_path = os.path.join(args.texture_file_directory, file)
            data_dict['file_path'] = file_path
            data_dict['filename'] = file
            data_dict['stimulus_duration'] = args.stimulus_duration
            data_dict['sample_frequency'] = args.sample_frequency
            data_dict['path_points'] = path_points
            data_dict['index'] = i
            data_dict['bounds'] = [args.x_upperbound, args.y_upperbound]
            data_dict['pins_per_mm'] = args.pins_per_mm
            data_dict['depth'] = args.depth
            data_dict['theta'] = args.theta
            data_dict['id'] = f"{i}_{file}" 
            path_list.append(data_dict)

    with open(os.path.join(args.results_folder, "info.json"), "w") as f:
        json.dump(path_list, f)
    args.num_child_threads = min(len(path_list), args.num_child_threads)
    items_per_child, remainder = divmod(len(path_list), args.num_child_threads)
    split_list = [path_list[i: i + items_per_child] 
                  for i in range(0, len(path_list), items_per_child)]
    if remainder > 0:
        split_list[-2].extend(split_list[-1])
        del split_list[-1]
    process_names = [f'P{i}' for i in range(args.num_child_threads)]
    processes = []
    for i in range(len(process_names)):
        process = multiprocessing.Process(target=child_process, args=(a, args.results_folder, process_names[i], split_list[i]))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()



if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Script for generating sample coordinates along a path of specified length")

    # Add arguments to the parser
    parser.add_argument("--texture_file_directory", type=str, help="Directory to get textures from")
    parser.add_argument("--results_folder", type=str, help="Directory to store results")
    parser.add_argument("--num_child_threads", default=os.cpu_count(), type=int, help="Number of processes to run")
    parser.add_argument("--texture_count", type=int, help="The number of textures to sample data from")
    parser.add_argument("--stimulus_duration", type=float, help="The duration for which each stimulus persists")
    parser.add_argument("--sample_frequency", type=float, help="The sampling rate for stimuli (in Hz)")
    parser.add_argument("--samples_per_texture", type=int, help="The number of samples to be generated for each texture")
    parser.add_argument("--x_upperbound", type=int, help="Upper bound for the maximum x coordinate")
    parser.add_argument("--y_upperbound", type=int, help="Upper bound for the maximum y coordinate")
    parser.add_argument("--distance", type=float, help="Required length of line")
    parser.add_argument("--pins_per_mm", type=float, help="Stimulus pin density")
    parser.add_argument("--depth", type=float, help="Indentation depth")
    parser.add_argument("--theta", type=float, help="Rotation of finger")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args)

