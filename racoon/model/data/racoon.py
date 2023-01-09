import torch
from PIL import Image
import os
from racoon import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def reverse_transform():
    transforms = []
    transforms.append(T.ConvertImageDtype(torch.uint8))
    return T.Compose(transforms)

class RaccoonDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms,labels):
        self.root = root
        self.transforms = transforms
        self.labels = labels
        self.labels['class_idx'] =  self.labels['class'].apply(lambda x : 2 if x=='miguel' else 1)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
        self.imgs = [img_path for img_path in self.imgs if img_path in self.labels['filename'].values]
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "train", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
    
        # instances are encoded as different colors
        obj_ids = self.labels.loc[self.labels['filename']==self.labels['filename'][idx]]
  
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(obj_ids[['xmin', 	'ymin', 	'xmax', 	'ymax']].values, dtype=torch.float32)
        # there is only one class
        
        labels = torch.tensor(self.labels['class_idx'].values, dtype=torch.int64)

        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)
