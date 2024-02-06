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
        # self.labels should be like ['apple', 'keyboard', 'sponge'], but we will use [0, 2, 5] because of pytorch
        self.labels = []
        self.current_label = 0
        label_dict = []
        for filename in self.filenames:
            basename = os.path.basename(filename)
            blocks = basename.split('_')
            
            if len(blocks) == 5:
                label = blocks[0]  # because basename is "sponge_10_4_34_crop.png"      
            else:
                label = blocks[0] + ' ' + blocks[1] #because some base names are like food_can_13_1_200_crop.png

            if label in label_dict:
                self.labels.append(label_dict.index(label))
                
            else:
                label_dict.append(label)
                self.labels.append(self.current_label)
                self.current_label += 1
                

        # print(self.filenames[0:1])
        # print(self.labels[0:15])
        # print(len(self.label_dict))
        # print(self.label_dict)
                
        # filenames ['/home/jose/Desktop/rgbd-dataset/bowl/bowl_3/bowl_3_4_4_crop.png', '/home/jose/Desktop/rgbd-dataset/food_bag/food_bag_8/food_bag_8_4_138_crop.png']
        # labels [0, 1, 2, 2, 3, 4, 5, 6, 7, 5, 8, 9, 1, 5, 10]
        # label_dict lengh 51
        # label_dict ['bowl', 'food bag', 'orange', 'toothbrush', 'food can', 'onion', 'lightbulb', 'bell pepper', 'sponge', 'potato', 'banana', 'lemon', 'soda can', 'peach', 'food box', 'notebook', 'kleenex', 'flashlight', 'stapler', 'keyboard', 'glue stick', 'cap', 'marker', 'comb', 'instant noodles', 'lime', 'plate', 'dry battery', 'cell phone', 'toothpaste', 'food cup', 'garlic', 'apple', 'coffee mug', 'water bottle', 'hand towel', 'mushroom', 'scissors', 'pliers', 'tomato', 'food jar', 'calculator', 'pear', 'shampoo', 'rubber eraser', 'ball', 'camera', 'pitcher', 'greens', 'cereal box', 'binder']
        exit(0)

        #TODO Maybe add some other transformations to reduce overfiting
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
