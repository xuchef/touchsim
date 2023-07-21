from PIL import Image
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from typing import Literal

import warnings
warnings.filterwarnings("ignore")


class DisplacementMap:
    """A representation of a textured surface, which is defined by control
    points, that can then be sampled across its area.
    """

    def __init__(self, filename: str, size=(None, None), max_height=20):
        """Initializes a DisplacementMap object."""
        self.size = size
        self.filename = filename
        self.max_height = max_height
        self.bitmap = self._get_bitmap(self.filename, self.size, self.max_height)
        self.interpolator = RegularGridInterpolator([np.arange(i) for i in self.bitmap.shape], self.bitmap,
                                                    bounds_error=False, fill_value=None, method="linear")

    def height_at(self, x: float, y: float) -> float:
        return self.interpolator((x, y))

    def visualize(self, coords: tuple[float, float] = None,
                  show_original_points=False, surface_type: Literal["square", "triangular", "mesh"] = "triangular",
                  full_screen=False, resolution=100) -> None:
        fig = plt.figure(figsize=(10, 10))
        plt.subplots_adjust(bottom=0, left=0, right=1, top=1)
        ax = fig.add_subplot(projection="3d", computed_zorder=False)  # WHAT THE HECK MATPLOTLIB!!!!
        ax.set_zlim(0, 255)
        ax.axis("off")

        if show_original_points:
            x, y = np.meshgrid(np.arange(self.bitmap.shape[0]), np.arange(self.bitmap.shape[1]))
            ax.scatter(x.ravel(), y.ravel(), self.bitmap.ravel(), s=10, c='k', label='data')

        xx, yy = np.linspace(0, self.bitmap.shape[0], resolution), np.linspace(0, self.bitmap.shape[1], resolution)
        X, Y = np.meshgrid(xx, yy, indexing="xy")

        if surface_type == "mesh":
            ax.plot_wireframe(X, Y, self.height_at(X, Y), alpha=0.4, cmap="viridis")
        if surface_type == "square":
            ax.plot_surface(X, Y, self.height_at(X, Y), alpha=0.8, cmap="viridis", edgecolor="none")
        if surface_type == "triangular":
            ax.plot_trisurf(X.ravel(), Y.ravel(), self.height_at(X, Y).ravel(),
                            cmap="viridis", edgecolor="none")

        if coords is not None:
            ax.scatter(*coords, self.height_at(*coords), c="m", marker=7, s=100, zorder=3)

        if full_screen:
            plt.get_current_fig_manager().full_screen_toggle()

        plt.show()

    def _get_bitmap(self, filename, size, max_height) -> np.ndarray[float]:
        image = Image.open(filename)
        image = image.resize(self._scale_dimensions(image.size, size))
        bitmap = np.array(image)
        bitmap = np.interp(bitmap, [0, 255], [0, max_height])

        return bitmap

    def _scale_dimensions(self, old: tuple[float, float], new: tuple[float, float]):
        """Scales dimensions accordingly if a dimension is None"""
        if new[0] is None and new[1] is None:
            return old
        if new[0] is None:
            new[0] = int(old[0] * new[1] / old[1])
        if new[1] is None:
            new[1] = int(old[1] * (new[0] / old[0]))
        return new


if __name__ == "__main__":
    texture = DisplacementMap("../textures/rock.jpg", size=(10, 10))
    texture.visualize((3, 4))
