import os.path
from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import zarr
from batchgenerators.utilities.file_and_folder_operations import isfile, load_json, save_json, split_path, join

class ZarrIO(BaseReaderWriter):

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        Reads a sequence of images and returns a 4d (!) np.ndarray along with a dictionary. The 4d array must have the
        modalities (or color channels, or however you would like to call them) in its first axis, followed by the
        spatial dimensions (so shape must be c,x,y,z where c is the number of modalities (can be 1)).
        Use the dictionary to store necessary meta information that is lost when converting to numpy arrays, for
        example the Spacing, Orientation and Direction of the image. This dictionary will be handed over to write_seg
        for exporting the predicted segmentations, so make sure you have everything you need in there!

        IMPORTANT: dict MUST have a 'spacing' key with a tuple/list of length 3 with the voxel spacing of the np.ndarray.
        Example: my_dict = {'spacing': (3, 0.5, 0.5), ...}. This is needed for planning and
        preprocessing. The ordering of the numbers must correspond to the axis ordering in the returned numpy array. So
        if the array has shape c,x,y,z and the spacing is (a,b,c) then a must be the spacing of x, b the spacing of y
        and c the spacing of z.

        In the case of 2D images, the returned array should have shape (c, 1, x, y) and the spacing should be
        (999, sp_x, sp_y). Make sure 999 is larger than sp_x and sp_y! Example: shape=(3, 1, 224, 224),
        spacing=(999, 1, 1)

        For images that don't have a spacing, set the spacing to 1 (2d exception with 999 for the first axis still applies!)

        :param image_fnames:
        :return:
            1) a np.ndarray of shape (c, x, y, z) where c is the number of image channels (can be 1) and x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (3, 1, 224, 224) for RGB image).
            2) a dictionary with metadata. This can be anything. BUT it HAS to include a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)
        """

        ending = '.' + image_fnames[0].split('.')[-1]
        assert ending.lower() == '.zarr', f'Ending {ending} not supported by {self.__class__.__name__}'

        images = []
        for f in image_fnames:
            zarr_store = zarr.open(f, mode='r')
            spacing = zarr_store.attrs.get("spacing", [1.0, 1.0, 1.0])
            image = zarr_store['neurofilament'][:]
            if image.ndim != 3:
                raise RuntimeError(f"Only 3D images are supported! File: {f}")
            images.append(image[None]) #None is an alias for NP.newaxis. It creates an axis with length 1

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        if spacing == [1.0,1.0,1.0]:
            print('WARNING spacing is (1, 1, 1). This probably happens because there is no spacing in zarr metadata')

        return np.vstack(images, dtype=np.float32, casting='unsafe'), {'spacing': spacing}


    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """
        Same requirements as BaseReaderWriter.read_image. Returned segmentations must have shape 1,x,y,z. Multiple
        segmentations are not (yet?) allowed

        If images and segmentations can be read the same way you can just `return self.read_image((image_fname,))`
        :param seg_fname:
        :return:
            1) a np.ndarray of shape (1, x, y, z) where x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (1, 1, 224, 224) for 2D segmentation).
            2) a dictionary with metadata. This can be anything. BUT it HAS to include a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)
        """
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        """
        Export the predicted segmentation to the desired file format. The given seg array will have the same shape and
        orientation as the corresponding image data, so you don't need to do any resampling or whatever. Just save :-)

        properties is the same dictionary you created during read_images/read_seg so you can use the information here
        to restore metadata

        IMPORTANT: Segmentations are always 3D! If your input images were 2d then the segmentation will have shape
        1,x,y. You need to catch that and export accordingly (for 2d images you need to convert the 3d segmentation
        to 2d via seg = seg[0])!

        :param seg: A segmentation (np.ndarray, integer) of shape (x, y, z). For 2D segmentations this will be (1, y, z)!
        :param output_fname:
        :param properties: the dictionary that you created in read_images (the ones this segmentation is based on).
        Use this to restore metadata
        :return:
        """
        zarr_store = zarr.open(output_fname, mode="a")

        segmentation = zarr_store.create(
            'segmentation',
            shape=seg.shape,
            dtype=seg.dtype,
            compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=zarr.Blosc.SHUFFLE)
        )

        segmentation[:] = seg[:, :, :]

        zarr_store.attrs.update({
            'spacing': properties['spacing']
        })
