import ast
import os, glob, warnings, cv2, nrrd
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import *

# import pandas as pd, random, pickle
# from sklearn.model_selection import KFold
# import scipy.ndimage.measurements as measure

# import SimpleITK as sitk
# np.random.seed(7)
# ROIS = ["External", "GTVp", "LCTVn", "RCTVn", "Brainstem", "Esophagus",
#         "Larynx", "Cricoid_P", "OpticChiasm", "Glnd_Lacrimal_L",
#         "Glnd_Lacrimal_R", "Lens_L", "Lens_R", "Eye_L", "Eye_R",
#         "Nrv_Optic_L", "Nrv_Optic_R", "Parotid_L", "Parotid_R",
#         "SpinalCord", "Mandible_Bone", "Glnd_Submand_L",
#         "Glnd_Submand_R", "Cochlea_L", "Cochlea_R", "Lips",
#         "Spc_Retrophar_R", "Spc_Retrophar_L", "BrachialPlex_R",
#         "BrachialPlex_L", "BRAIN", "OralCavity", "Musc_Constrict_I",
#         "Musc_Constrict_S", "Musc_Constrict_M"]

# def getROIOrder(custom_order=None, rois=ROIS):
#     # ideally this ordering has to be consistent to make inference easy...
#     if custom_order is None:
#         order_dic = {roi:i for i, roi in enumerate(ROIS)}
#     else:
#         roi_order = [rois[i] for i in custom_order]
#         order_dic = {roi:i for i, roi in enumerate(roi_order)}
#     return order_dic

# testing the getROIOrder function > can use this to rearange ordering of mask loading...
# this was the custom ordering used to develop the model in question...
# custom_order = [4,20,5,6,22,17,18,25,26,30,31,11,12,13,14,15,16,8,27]
# v_ = getROIOrder(custom_order=custom_order)

class LoadPatientVolumes(Dataset):
    def __init__(self, folder_data, data_config, tag="neck", transform=None,
                 cache_dir="/cluster/projects/radiomics/Temp/joe/scratch_1",
                 mode="train"): #"/h/jmarsilla/scratch"
        """
        This is the class that our Dataloader object
        will take to create batches for training.
        param: image_data : This is a dataframe with image path,
                            and folder location of structures
        param: data_config: Dictionary with all loading and
        """
        self.data = folder_data
        if mode == "test":
            data_config["data_path"] = "/cluster/projects/radiomics/Temp/joe/RADCURE_VECTOR_UPDATE_TEST/"
        self.config = data_config
        self.transform=transform
        self.cache_dir = cache_dir
        self.tag = tag

    def __len__(self):
        return len(self.data)

    def load_data(self, idx):
        """
        Loading of the data...
        """
        self.patient = str(self.data.iloc[idx][0])
        if self.config["data_path"][-1]!="/":
            self.config["data_path"]+= "/"
        self.img_path = self.config["data_path"] + f'{self.patient}/CT_IMAGE.nrrd'
        self.structure_path = self.config["data_path"] + f'{self.patient}/structures/'
        # can write your own custom finction to load in structures here...
        # assumes directory structure where patient name is enclosing folder...
        custom_order = self.config['roi_order']
        self.window = self.config["window"]
        self.crop_width = self.config["crop_width"]
        self.order_dic = getROIOrder(tag=self.tag)
        self.oars = list(self.order_dic.keys())
        self.load_nrrd()

    def load_nrrd(self):
        # load image using nrrd...
        warnings.warn('Using nrrd instead of sitk.')
        self.img, header = nrrd.read(self.img_path)
        self.img = self.img # .transpose(2,0,1)
        shape = self.img.shape
        # header = nrrd.read_header(self.img_path)
        # Loading with SITK...
        # mask = sitk.ReadImage(img_path)
        # img = sitk.ReadImage(img_path)
        # self.img = sitk.GetArrayFromImage(img)
        # self.mask = sitk.GetArrayFromImage(mask)
        # Generate MASK according to order defined in config
        # Can also define custom_order in data_config...
        # mask_path should be list to folder of structures
        # names should match the labels given in ROI list above...
        if self.structure_path[-1] != '/':
            self.structure_path += '/'
        mask_paths = glob.glob(self.structure_path + "*")
        # version 2 ... for other losses one hot encoding might be required
        # self.mask = []
        # this will load in masks and set them to class value of order_dic
        # can be modified in target adaptive loss or be used to condition network to missing labels.
        name_ = f"/{self.patient}_{self.tag}_mask.nrrd"
        cache_file = self.cache_dir + name_.lower().replace("-", "_")

        if os.path.isfile(cache_file):
            mask = nrrd.read(cache_file)
            header = mask[1]
            self.mask = mask[0]
            # header = nrrd.read_header(cache_file)
            self.count = ast.literal_eval(header["counts"]) # ["Count"]
            self.count = np.array([int(float(a)) for a in self.count])
            warnings.warn(f"Loaded {self.patient} from cache.")
        else:
            # version 1 ...
            # set empty mask... useful for weightedtopkCE loss
            self.mask = np.zeros(shape)
            # meta = {}
            self.count = np.zeros(len(self.oars)+1)
            self.count[0] = 1
            # only the case if EXTERNAL NOT included in cases...
            for path in mask_paths:
                oar = path.split('/')[-1].partition('.')[0]
                if oar in self.oars:
                    class_value = self.order_dic[oar]
                    mask = nrrd.read(path)
                    mask = mask[0] # .transpose(2,0,1)
                    warnings.warn(f"Loading {oar} for {self.patient} has {mask.shape} v. {shape}. Max val is {mask.max()} will be {class_value}.")
                    assert mask.shape == shape
                    # version 1 > not one hot encoded
                    self.mask[mask>0] = class_value
                    self.count[class_value] = 1
            # save file to scratch folder...
            # meta["img_header"] = header
            # meta["count"] = self.count

            nrrd.write(cache_file, self.mask, header={"counts": list(self.count)}) #, compression_level=9)
        
        if shape[0]<self.window*2:
            warnings.warn(f'Padding {self.patient} z shape of {shape[0]} to {self.window*2}')
            difference = self.window*2 - shape[0]
            a = difference//2
            diff = difference-a
            npad = ((a, diff), (0, 0), (0, 0))
            self.img = np.pad(self.img, pad_width=npad, mode='constant', constant_values=self.img.min())
            self.mask = np.pad(self.mask, pad_width=npad, mode='constant', constant_values=0)
            warnings.warn(f'NEW size is {self.img.shape},')
            # this is how we would pad the same tensor using pytorch if was a tensor...
            # pad_ = (0,0,0,0,a,diff)
            # npad is a tuple of (n_before, n_after) for each dimension
            # self.img = F.pad(self.img, pad_, "constant", self.img.min())
            # self.mask = F.pad(self.mask, pad_, "constant", 0)

    def __getitem__(self, idx):

        self.load_data(idx)
        # if self.transform is not None:
        warnings.warn(f"{self.mask.shape} mask vs {self.img.shape} img")
        if self.mask.max() > 0:
            try:
                self.img, self.mask = self.transform(self.img.copy(), self.mask.copy())
                assert (self.img.shape[1], self.img.shape[2]) == (self.crop_width, self.crop_width)
            except Exception as e:
                # if shape is not 2x window size we need to recrop...
                warnings.warn(str(e))
                raise Exception(f"Please check mask for folder {self.patient}.")
        else:
            warnings.warn("No masks for segmentation task.")
            # use image twice...
            self.img, self.mask = self.transform(self.img.copy(), self.img.copy())
        
        try:
            assert self.mask.max() > 0
        except Exception:
            warnings.warn(f"Cropped out all class values for {self.patient}...")
            
        img = torch.from_numpy(self.img).type(torch.FloatTensor)
        mask = torch.from_numpy(self.mask).type(torch.LongTensor)
        count = torch.from_numpy(self.count).type(torch.LongTensor)
        return img, mask, count