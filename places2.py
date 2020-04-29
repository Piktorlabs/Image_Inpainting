import random
import torch
from PIL import Image
from glob import glob


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        # print("split",split)

        # use about 8M images in the challenge dataset
        if split == 'train':
            # print("inside if", img_root, split)
            self.paths = glob('{:s}/train/**/*.jpg'.format(img_root),
                              recursive=True)
        else:
            # print("inside else", img_root, split)
            self.paths = glob('{:s}/{:s}/*'.format(img_root, split))
        
        # print('{:s}/train/**/*.jpg'.format(img_root))

        self.mask_paths = glob('{:s}/*.jpg'.format(mask_root))
        self.N_mask = len(self.mask_paths)
        # print("p",self.N_mask,  self.paths , self.mask_paths, '{:s}/*.jpg'.format(mask_root))
            


    def __getitem__(self, index):
        # print(self.paths,"--", index, self.paths[index])
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        # mask = Image.open(self.mask_paths[index]) #to match image and mask
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
