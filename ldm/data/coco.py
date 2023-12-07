from PIL import Image
import os
import json
import albumentations
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class CocoBase(Dataset): 
    def __init__(self, config=None, size=None, interpolation="bicubic", random_crop=False, crop_size=None):
        self.split = self.get_split()
        self.root_dir = "/hub_data2/inho/data/coco"

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
            json_cocotrain = os.path.join(self.root_dir, 'annotations', 'captions_train2017.json')
            f = open(json_cocotrain)
            coco_data = json.load(f)['annotations']

            for item in coco_data:
                image_path = os.path.join(self.root_dir,'train2017', (12-len(str(item['image_id'])))*"0"+ str(item['image_id'])+'.jpg')
                data.append({'image_path': os.path.join(image_path), 'caption_ori': item['caption'], 'caption_prompt': 'A clean photo of '+item['caption'],'image_id':str(item['image_id'])})

        elif self.split == 'val':
            json_cocotrain = os.path.join(self.root_dir, 'annotations', 'captions_val2017.json')
            f = open(json_cocotrain)
            coco_data = json.load(f)['annotations']
            sorted_data = sorted(coco_data, key=lambda d: d['image_id'])

            bef_id = None
            for item in sorted_data:
                image_id = (12-len(str(item['image_id'])))*"0"+ str(item['image_id'])
                if bef_id == image_id:
                    continue
                else:
                    bef_id = image_id
                image_path = os.path.join(self.root_dir,'val2017', image_id+'.jpg')
                data.append({'image_path': os.path.join(image_path), 'caption_ori': item['caption'], 'caption_prompt': 'A clean photo of '+item['caption'], 'image_id':image_id})

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

        if self.split == 'train':
            if random.rand() < 0.5:
                example= {"image" : (image / 127.5 - 1.0).astype(np.float32), "caption_ori": self.data[i]['caption_ori'], "caption_prompt": self.data[i]['caption_prompt'], "image_id":self.data[i]['image_id']}
            else:
                example= {"image" : (image / 127.5 - 1.0).astype(np.float32), "caption_ori": self.data[i]['caption_prompt'], "caption_prompt": self.data[i]['caption_prompt'], "image_id":self.data[i]['image_id']}
                
        elif self.split == 'val':
            example= {"image" : (image / 127.5 - 1.0).astype(np.float32), "caption_ori": self.data[i]['caption_ori'], "caption_prompt": self.data[i]['caption_prompt'], "image_id":self.data[i]['image_id']}


        return example


class CocoTrain(CocoBase):
    def __init__(self, config=None, size=None, random_crop=True, interpolation="bicubic", crop_size=None):
        super().__init__(config=config, size=size,
                          interpolation=interpolation)

    def get_split(self):
        return "train"


class CocoValidation(CocoBase):
    def get_split(self):
        return "val"