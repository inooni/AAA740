from PIL import Image
import os
import json
import albumentations
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LaionBase(Dataset): 
    def __init__(self, config=None, size=None, interpolation="bicubic", random_crop=False, crop_size=None):
        self.split = self.get_split()
        self.root_dir = "/hub_data2/inho/data/laion_subset_m"

        self.data = self.load_data()
        self.size = size

        if crop_size is None:
            self.crop_size = size if size is not None else None
        else:
            self.crop_size = crop_size

        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)

        if self.crop_size is not None:
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            self.preprocessor = self.cropper

    def load_data(self):
        
        data = []

        if self.split == 'train':
            json_orig_train = os.path.join(self.root_dir, 'annotations/orig', 'train.json')
            json_template1_train = os.path.join(self.root_dir, 'annotations/template1', 'train.json')
            
            f_o = open(json_orig_train)
            f_t = open(json_template1_train)
            laion_data_o = sorted(json.load(f_o), key=lambda d: d['image_id'])
            laion_data_t = sorted(json.load(f_t), key=lambda d: d['image_id'])

            for i in range(len(laion_data_o)):

                assert laion_data_o[i]['image_path'] == laion_data_t[i]['image_path']

                image_path = os.path.join(self.root_dir[:-19],laion_data_o[i]['image_path'])
                data.append({'image_path': os.path.join(image_path), 'caption_ori': laion_data_o[i]['text'], 'caption_prompt': 'A clean photo of '+laion_data_o[i]['text'], 'image_id': str(laion_data_o[i]['image_id'])})


        elif self.split == 'val':
            json_orig_train = os.path.join(self.root_dir, 'annotations/orig', 'test.json')
            json_template1_train = os.path.join(self.root_dir, 'annotations/template1', 'test.json')
            
            f_o = open(json_orig_train)
            f_t = open(json_template1_train)
            laion_data_o = sorted(json.load(f_o), key=lambda d: d['image_id'])
            laion_data_t = sorted(json.load(f_t), key=lambda d: d['image_id'])

            for i in range(len(laion_data_o)):

                assert laion_data_o[i]['image_path'] == laion_data_t[i]['image_path']

                image_path = os.path.join(self.root_dir[:-19],laion_data_o[i]['image_path'])
                data.append({'image_path': os.path.join(image_path), 'caption_ori': laion_data_o[i]['text'], 'caption_prompt': 'A clean photo of '+laion_data_o[i]['text'], 'image_id': str(laion_data_o[i]['image_id'])})

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i): 

        image_path = self.data[i]['image_path']        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.uint8)

        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]

        if self.size is not None:
            image = self.preprocessor(image=image)["image"]        

        example= {"image" : (image / 127.5 - 1.0).astype(np.float32), "caption_ori": self.data[i]['caption_ori'], "caption_prompt": self.data[i]['caption_prompt'], 'image_id':self.data[i]['image_id']}
        return example


class LaionTrain(LaionBase):
    def __init__(self, config=None, size=None, random_crop=True, interpolation="bicubic", crop_size=None):
        super().__init__(config=config, size=size,
                          interpolation=interpolation)

    def get_split(self):
        return "train"


class LaionValidation(LaionBase):
    def get_split(self):
        return "val"