#!/usr/bin/env python3


import os
import torch
from torchvision import transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames):
        self.filenames = filenames
        self.number_of_images = len(self.filenames)

        # Compute the corresponding labels
        # self.labels should be like ['cat', 'dog', 'cat'], but we will use [1, 0, 1] because of pytorch
        self.labels = []
        self.current_label = 0
        self.label_dict = []
        for filename in self.filenames:
            basename = os.path.basename(filename)
            blocks = basename.split('.')
            label = blocks[0]  # because basename is "cat.2109.jpg"

            if label in self.label_dict:
                self.labels.append( self.label_dict.index(label))
                
            else:
                self.label_dict.append(label)
                self.labels.append(self.current_label)
                self.current_label += 1
                

        print(self.filenames[0:10])
        print(self.labels[0:10])
        print(self.label_dict [0:10])
        # filenames ['/home/jose/Desktop/train/dog.12026.jpg', '/home/jose/Desktop/train/cat.10739.jpg', '/home/jose/Desktop/train/dog.5728.jpg']
        # labels ['0', '1', '0']

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        # must return the size of the data
        return self.number_of_images

    def __getitem__(self, index):
        # Must return the data of the corresponding index

        # Load the image in pil format
        filename = self.filenames[index]
        pil_image = Image.open(filename)

        # Convert to tensor
        tensor_image = self.transforms(pil_image)

        # Get corresponding label
        label = self.labels[index]

        return tensor_image, label
