import warnings
import touchsim as ts  # import touchsim package
from touchsim.plotting import plot  # import in-built plotting function
import numpy as np
import holoviews as hv  # import holoviews package for plots and set some parameters

from touchsim.surface import Surface, hand_surface
import matplotlib.pyplot as plt
import sys


warnings.simplefilter("ignore")
hv.extension("matplotlib")
np.set_printoptions(threshold=sys.maxsize)


def print_first_region(obj=hand_surface, region="D2d_t"):
    idx = obj.tag2idx(region)

    amin = np.min(obj.bbox_min[idx], axis=0).astype(int)
    amax = np.max(obj.bbox_max[idx], axis=0).astype(int)

    # empty bounding box
    im = np.nan * np.zeros(tuple((amax-amin+1)[::-1].tolist()))

    # fill in
    for i in idx:
        # print(obj._coords[i][:, 1])
        # im[obj._coords[i][:, 1]-amin[1], obj._coords[i][:, 0]-amin[0]] = 3
        # im[obj._coords[i] - amin] = 3
        x = obj._coords[i] - amin
        im[x[:, 1], x[:, 0]] = 3

    im = np.flipud(im)
    for row in im:
        for i in row:
            print("X" if i == 3 else " ", end="")
        print()

    # print(obj.hand2pixel(amin), obj.hand2pixel(amax))
    # print(obj.hand2pixel(amin), obj.hand2pixel(amax))

    # boundary = obj.boundary[idx[0]]

    # boundary = boundary.tolist()
    # for j in range(int(amax[1] - amin[1]) + 1):
    #     for i in range(int(amax[0] - amin[0]) + 1):
    #         if [amin[0] + i, amax[1] - j] in boundary:
    #             print("X", end="")
    #         else:
    #             print(" ", end="")
    #     print()

    # fig, ax = plt.subplots()
    # ax.plot(boundary[:, 0], boundary[:, 1])
    # plt.show()


def index():
    obj = hand_surface

    region = "D2d_t"
    idx = obj.tag2idx(region)

    amin = np.min(obj.bbox_min[idx], axis=0)
    amax = np.max(obj.bbox_max[idx], axis=0)
    print(amin, amax)
    # print(obj.boundary[idx[0]])


def visualize_circle():
    mr = hv.renderer('matplotlib')

    a = ts.affpop_hand()

    s = ts.stim_indent_shape(ts.shape_circle(hdiff=0.5, pins_per_mm=2, radius=2, center=(0, 12)), ts.stim_ramp(len=0.1))
    mr.show(plot(region='D2', coord=10) * plot(s, spatial=True))

    # r = a.response(s)
    # print(r)
    # mr.show(plot(r))
    # mr.show(plot(region='D2', coord=10) * plot(r, spatial=True, scaling_factor=.1))


def visualize_bar():
    mr = hv.renderer('matplotlib')

    a = ts.affpop_hand()

    s = ts.stim_indent_shape(ts.shape_bar(hdiff=0.5, width=5, height=5,
                             center=(0, 0), pins_per_mm=1), ts.stim_ramp(len=0.1))
    mr.show(plot(region='D2d_t', coord=10) * plot(s, spatial=True))


def visualize_finger_indent():
    mr = hv.renderer('matplotlib')

    a = ts.affpop_hand()

    s = ts.stim_indent_shape(ts.shape_hand_region(region="D2d_t", pins_per_mm=1), ts.stim_ramp(len=0.1))
    # print(len(s.trace))
    mr.show(plot(region='D2d_t', coord=10) * plot(s, spatial=True))

    # r = a.response(s)
    # print(r)
    # mr.show(plot(r))
    # mr.show(plot(region='D2', coord=10) * plot(r, spatial=True, scaling_factor=.1))


def texture_stim():
    mr = hv.renderer('matplotlib')

    a = ts.affpop_hand()

    disp_map = ts.texture.DisplacementMap(filename="./textures/tile.jpg", size=(500, 500), max_height=1)

    num_samples = 600

    x = np.full(num_samples, disp_map.shape[0] / 4)
    y = np.linspace(disp_map.shape[1] / 4, disp_map.shape[1] * 3 / 4, num_samples)
    path = np.column_stack((x, y))

    loc = np.zeros((1, 2))

    pins = ts.shape_hand_region(region="D2d_t", pins_per_mm=1)
    s = ts.stim_texture_indextip(disp_map, path, pins, (0, 0))

    # s = ts.stim_indent_shape(ts.shape_hand_region(region="D2d_t", pins_per_mm=1), ts.stim_ramp(len=0.1))
    # s = ts.stim_texture_indextip(disp_map, path, pins)
    # print(len(s.trace))
    # mr.show(plot(s))
    # mr.show(plot(region='D2d_t', coord=10) * plot(s, spatial=True))

    r = a.response(s)
    # print(r)
    mr.show(plot(r))
    # mr.show(plot(region='D2', coord=10) * plot(r, spatial=True, scaling_factor=.1))


if __name__ == "__main__":
    # print_first_region()
    # visualize_circle()
    # visualize_bar()
    # ts.generators.shape_hand_region(region="D2d_t", pins_per_mm=2)
    # visualize_finger_indent()
    texture_stim()

    # x = np.array([[1, 2], [3, 4], [5, 6]])
    # print(hi(x))
