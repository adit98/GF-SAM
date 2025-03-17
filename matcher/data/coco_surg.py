r""" Surg few-shot segmentation dataset """
import os
import json

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
from pycocotools import mask as maskUtils

class DatasetCOCO_surg(Dataset):
    def __init__(self, datapath, benchmark, fold, transform, split, shot, use_original_imgsize):
        self.fold = fold
        self.nfolds = 3
        self.benchmark = 'coco'
        self.shot = shot
        self.base_path = os.path.join(datapath, benchmark)
        if benchmark == 'endoscapes':
            self.train_path = os.path.join(self.base_path, 'train_seg')
            self.test_path = os.path.join(self.base_path, 'test_seg')
            self.nclass = 7
        elif benchmark == 'hyperkvasir':
            self.train_path = os.path.join(self.base_path, 'train', 'images')
            self.test_path = os.path.join(self.base_path, 'test', 'images')
            self.nclass = 2
        elif benchmark == 'cadis':
            self.train_path = self.base_path
            self.test_path = self.base_path
            self.nclass = 8
        elif benchmark == 'c8k':
            self.train_path = self.base_path
            self.test_path = self.base_path
            self.nclass = 9

        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.build_json_metadata()

    def build_class_ids(self):
        class_ids = [x for x in range(self.nclass)]
        return class_ids

    def build_json_metadata(self):
        # Load JSON metadata for train and test sets.
        with open(f'{self.base_path}/annotations/train.{self.shot}shot_{self.fold}.annotation_coco.json') as f:
            self.train_json_data = json.load(f)

        with open(f'{self.base_path}/annotations/test.annotation_coco.json') as f:
            self.test_json_data = json.load(f)

        # Build image id to image info mapping.
        self.test_images = {img['id']: img for img in self.test_json_data['images']}
        self.train_images = {img['id']: img for img in self.train_json_data['images']}

    def __len__(self):
        # Define the dataset length as the number of query annotations.
        return len(self.test_json_data['annotations'])

    def __getitem__(self, idx):
        # --- Query processing ---
        # Retrieve the query annotation (each annotation corresponds to one instance)
        query_ann = self.test_json_data['annotations'][idx]
        query_img_info = self.test_images[query_ann['image_id']]
        query_file_name = query_img_info['file_name']
        #base_name, _ = os.path.splitext(query_file_name)
        base_name = query_file_name
        query_img_path = os.path.join(self.test_path, query_file_name)

        # Load query image and store its original size (width, height)
        query_img = Image.open(query_img_path).convert("RGB")
        org_qry_imsize = query_img.size

        # Decode the RLE mask for the query instance.
        query_mask_np = maskUtils.decode(query_ann['segmentation'])
        query_mask_tensor = torch.tensor(query_mask_np).float()

        # Apply transformation to the query image if provided.
        if self.transform is not None:
            query_img = self.transform(query_img)

        # Resize the query mask to match the transformed image if not using the original image size.
        if not self.use_original_imgsize:
            # Assume query_img is now a tensor with shape [C, H, W].
            new_size = query_img.shape[-2:]
            query_mask_tensor = F.interpolate(
                query_mask_tensor.unsqueeze(0).unsqueeze(0),
                size=new_size,
                mode='nearest'
            ).squeeze()

        # --- Support set processing ---
        # Build the support set: collect all training examples for the same object category.
        query_category = query_ann['category_id']
        support_imgs_list = []
        support_masks_list = []
        support_names_list = []

        for ann in self.train_json_data['annotations']:
            if ann['category_id'] == query_category:
                support_img_info = self.train_images[ann['image_id']]
                support_file_name = support_img_info['file_name']
                #support_base_name, _ = os.path.splitext(support_file_name)
                support_base_name = support_file_name
                support_img_path = os.path.join(self.train_path, support_file_name)

                # Load support image and decode its RLE mask.
                support_img = Image.open(support_img_path).convert("RGB")
                support_mask_np = maskUtils.decode(ann['segmentation'])
                support_mask_tensor = torch.tensor(support_mask_np).float()

                # Apply transformation if provided.
                if self.transform is not None:
                    support_img = self.transform(support_img)

                # Resize support mask if not using the original image size.
                if not self.use_original_imgsize:
                    new_size_support = support_img.shape[-2:]
                    support_mask_tensor = F.interpolate(
                        support_mask_tensor.unsqueeze(0).unsqueeze(0),
                        size=new_size_support,
                        mode='nearest'
                    ).squeeze()

                support_imgs_list.append(support_img)
                support_masks_list.append(support_mask_tensor)
                support_names_list.append(support_base_name)

        if len(support_imgs_list) > 0:
            support_imgs = torch.stack(support_imgs_list)
            support_masks = torch.stack(support_masks_list)
        else:
            support_imgs = torch.empty(0)
            support_masks = torch.empty(0)

        batch = {
            'query_img': query_img,
            'query_mask': query_mask_tensor,
            'query_name': base_name,
            'org_query_imsize': org_qry_imsize,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names_list,
            'class_id': torch.tensor(query_category)
        }

        return batch
