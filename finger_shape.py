from PIL import Image
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def autocrop(image_path, tolerance=0, x_buffer=0, y_buffer=0):
    # Open the image using Pillow
    image = Image.open(image_path)

    # Convert the image to grayscale
    black_and_white = image.convert('L')

    # Threshold the image to create a binary image (foreground and background)
    threshold_value = 255 - tolerance
    black_and_white = black_and_white.point(lambda p: p < threshold_value and 255)

    # Get the bounding box of the non-background region
    bbox = black_and_white.getbbox()

    # Expand the bounding box with the buffer
    buffer_box = (
        bbox[0] - x_buffer,
        bbox[1] - y_buffer,
        bbox[2] + x_buffer,
        bbox[3] + y_buffer
    )

    # Crop the image using the expanded bounding box
    cropped_image = image.crop(buffer_box)

    return cropped_image


def is_in_domain(x, y):
    return y > -0.5 * np.log(np.power(1 - np.power(x, 2) * 16/9, 1/2))


def f(x, y):
    """
    x: (-0.75, 0.75)
    y: (0, 3)
    """
    return np.exp(-2 * y) + (-np.sqrt(1 - np.power(x, 2) * 16/9) + 1)


def generate_finger(animate=False, res=100, surface_type="square", num_frames=20, add_to_latex=False):
    assert surface_type in ["squarel", "triangular"]
    plt.rcParams['savefig.dpi'] = 600
    # plt.rcParams['savefig.format'] = 'png'

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=0, left=0, right=0.9, top=1)
    ax = fig.add_subplot(projection='3d')
    ax.axis("off")
    ax.view_init(elev=-10, azim=-50)

    X = np.linspace(-0.75, 0.75, res)
    Y = np.linspace(0, 3, res)

    # demo data
    # X = np.linspace(-3, 3, 1000)
    # Y = np.linspace(-5, 5, 1000)

    # map to proper range of values
    # X = np.interp(X, [min(X), max(X)], [-0.75, 0.75])
    # Y = np.interp(Y, [min(Y), max(Y)], [0, 3])

    # construct meshgrid
    X, Y = np.meshgrid(X, Y)

    # filter domain
    X, Y = np.where(is_in_domain(X, Y), X, 0), np.where(is_in_domain(X, Y), Y, 0)

    # get output values
    Z = f(X, Y)

    # plot
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))

    if surface_type == "square":
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
    if surface_type == "triangular":
        ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), cmap="viridis", edgecolor="None")

    def update(i):
        angle = i / num_frames * 360
        print(f"frame {i}/{num_frames}", end="\r")
        ax.view_init(elev=-10, azim=angle)
        if surface_type == "square":
            surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
        if surface_type == "triangular":
            ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), cmap="viridis", edgecolor="None")
        return fig,

    if animate:
        angles = np.arange(num_frames)
        ani = FuncAnimation(fig, update, frames=angles, blit=True, interval=0)
        name = f"finger_res{res}_{surface_type}_{num_frames}"
        ani.save(name + ".gif", writer='ffmpeg', fps=120)
    else:
        image_path = name + ".png"
        plt.savefig(image_path)

        if add_to_latex:
            output_image = autocrop(input_image_path, tolerance=30, x_buffer=6*70, y_buffer=6*100)
            output_image.save(f"/Users/michaelxu/Documents/LaTeX/FingerEquation/{name}.png")


if __name__ == "__main__":
    generate_finger(animate=True, res=1000, surface_type="triangular", num_frames=360, add_to_latex=False)
