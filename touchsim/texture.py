from PIL import Image
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class Texture(object):
    """A representation of a textured surface, which comprises of control points
    defined by a displacement map image that can then be interpolated.
    """
    def __init__(self,**args):
        """Initializes a DisplacementMap object.
        
        Kwargs:
            filename (string): Filename (with optional path) to an image file
                containing texture displacement map (for example: 'textures/tile.jpg',
                default: None). Note that the image is automatically converted to
                greyscale (single channel luminance).
            size (array): Width and height of desired coordinate space in pixels
                (default: [None, None]). If both width and height are None, the
                size of the image is used as the coordinate space dimensions. If
                one of width or height is None, the missing value is autocomputed
                such that the aspect ratio of the original image is maintained.
                PIL bicubic resampling is used for resizing the image.
            max_height (float): The maximum displacement height (default: 1).
                The pixel values are linearly mapped from [0, 255] -> [0, max_height].
        """
        self.size = args.get('size',np.array([None, None]))
        self.filename = args.get('filename',None)
        self.max_height = args.get('max_height',1.)

        self.bitmap = image2bitmap(self.filename, self.size, self.max_height)
        self.interpolator = RegularGridInterpolator(
            points=[np.arange(i) for i in self.bitmap.shape],
            values=self.bitmap,
            method="cubic",
            bounds_error=False,
            fill_value=0)
        

    @property
    def shape(self):
        return self.bitmap.shape

    def height_at(self,location):
        """Calculates the height of the texture at a specificied location by 
        interpolating the displacement map control points.

        Args:
            location (array): Location to sample at (e.g. [3.1, 4.5]).

        Returns:
            Float representing the texture height at the specified location.
        """
        return self.interpolator(location)


def scale_dimensions(old,new):
    """Proportionally scales missing dimension values according to the aspect ratio of
    the old dimensions.
    """
    if new[0] is None and new[1] is None:
        return old
    if new[0] is None:
        new[0] = int(new[1] / old[1] * old[0])
    if new[1] is None:
        new[1] = int(new[0] / old[0] * old[1])
    return new

def image2bitmap(filename,size,max_height):
    """Converts image to greyscale, resizes it with bicubic resampling, and generates a
    bounded 2D bitmap from the luminance value.
    """
    image = Image.open(filename).convert('L')
    image = image.resize(scale_dimensions(image.size, size), Image.BICUBIC)
    bitmap = np.array(image)
    bitmap = np.interp(bitmap, [0, 255], [0, max_height])

    return bitmap
