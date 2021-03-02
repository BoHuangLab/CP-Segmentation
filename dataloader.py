import os

import numpy as np
import skimage
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torch

from psutil import virtual_memory

from imgfileutils import get_imgtype, get_metadata, get_metadata_ometiff
from apeer_ometiff_library import omexmlClass


class CellImagesDataset(Dataset):
    """Cell Images To Segment"""

    def __init__(self, parent_folder, metadata_path=None):
        """
        Args:
            config_file (string): Path to the xml file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.parent_folder = parent_folder
        self.img_paths_list = []
        self.metadata_path = metadata_path

        for root, dirs, files in os.walk(self.parent_folder):
            for file_name in files:
                if ('nuclei_mask' or 'cyto_mask') in root:
                    continue
                else:
                    # look for .czi, .ome.tiff, .tiff, .jpg, .png, .gif
                    imgtype = get_imgtype(file_name)

                    if imgtype is not None:
                        img_path = os.path.join(root, file_name)

                        if os.name == "nt":
                            img_path = "\\\\?\\" + os.path.abspath(img_path)

                        self.img_paths_list.append(img_path)

        if len(self.img_paths_list) == 0:
            raise TypeError("No images or Non-standard image files found.")

    def __len__(self):
        return len(self.img_paths_list)

    def __getitem__(self, idx):

        img_path = self.img_paths_list[idx]
        image = skimage.io.imread(img_path)

        if self.metadata_path is not None:
            tree = ET.parse(self.metadata_path)
            root = tree.getroot()
            omexml = ET.tostring(root, encoding=None, method="xml")
            omemd = omexmlClass.OMEXML(omexml)

            md = get_metadata_ometiff(img_path, omemd)

        elif self.metadata_path is None:
            # get metadata from image if no metadata file found
            md, additional_mdczi = get_metadata(img_path)

            # FUTURE IMPLEMENTATION if not .ome.tiff or .czi, check if metadata file in folder
            # if md == None:
            #
            #     parent_folder = os.path.dirname(img_path)
            #     for fname in os.listdir(parent_folder):
            #         if fname.endswith('.xml'):
            #             xml_path = os.path.join(parent_folder,fname)
            #             tree = ET.parse(xml_path)
            #             root = tree.getroot()
            #             omexml = ET.tostring(root, encoding=None, method='xml')
            #             omemd = omexmlClass.OMEXML(omexml)
            #             md = get_metadata_ometiff(img_path, omemd)
            #             break

        if md is not None:
            return img_path, image, md

        elif md is None:
            raise ValueError("No metadata found.")

    def get_file_sizes(self):
        file_size_list = [os.path.getsize(path) for path in self.img_paths_list]
        return file_size_list


class LoadImageBatch(Dataset):
    def __init__(self, parent_folder, metadata_path=None, device="cpu"):

        self.dataset = CellImagesDataset(
            parent_folder=parent_folder, metadata_path=metadata_path
        )
        
        self.device = device

        if "cpu" in str(self.device):
            self.total_free_memory = virtual_memory().available
            threshold = .375

        elif "cuda" in str(self.device):
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            self.total_free_memory = r - a  # free inside reserved
            threshold = .25

        dataset = CellImagesDataset(parent_folder, metadata_path=None)

        self.stop_indexes = []

        total_size = 0
        
        

        for i, file_size in enumerate(self.dataset.get_file_sizes()):

            total_size = total_size + file_size

            if total_size > threshold * self.total_free_memory:
                self.stop_indexes.append(i)
                total_size = 0

        print("Analyzing data in " + str(len(self.stop_indexes) + 1) + " batches...")

    def __len__(self):
        return len(self.stop_indexes) + 1

    def __getitem__(self, idx):

        batch = []

        if len(self.stop_indexes) == 0:

            for i in range(len(self.dataset)):
                output = self.dataset[i]
                if output is not None:
                    img_path, image, metadata = output

                    image = prep_image(image, metadata)
                    batch.append([image, metadata, img_path])

        elif len(self.stop_indexes) > 0:

            for i in range(self.stop_indexes[idx], self.stop_indexes[idx + 1]):
                output = self.dataset[i]
                if output is not None:
                    img_path, image, metadata = output
                    batch.append([image, metadata, img_path])

        return batch


def prep_image(image, metadata):

    SizeX = metadata["SizeX"]
    SizeY = metadata["SizeY"]
    SizeZ = metadata["SizeZ"]
    SizeC = metadata["SizeC"]

    if SizeZ > 1:
        if len(image.shape) == 3:
            image = np.array([image])

        if image.shape[1] != SizeC and image.shape[1] > 60:
            image = image.transpose(0, 3, 1, 2)

    elif SizeZ == 1:
        if len(image.shape) == 2:
            image = np.array([image])

        if image.shape[0] != SizeC and image.shape[0] > 60:
            image = image.transpose(2, 0, 1)

    else:
        raise ValueError("Check SizeC >=1 in Metadata")

    return image
