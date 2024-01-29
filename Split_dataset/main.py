#!/usr/bin/env python3


import glob
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil



def main():

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    data_path = '/home/jose/Desktop/rgbd-dataset/'                    
    files1 = os.listdir(data_path)
    #print(len(files1))
    image_filenames = []

    for i in range(len(files1)):
        file1_name = str(files1[i])
        #print(file1_name)
        files2 = os.listdir(data_path + file1_name +'/')
        for i in range(len(files2)):
            file2_name = files2[i]
            image_filenames = image_filenames + glob.glob(data_path + file1_name + '/' + file2_name + '/' + '*_crop.png')
            



    # Use a rule of 80% train, 20% validation    
    train_filenames, validation_filenames = train_test_split(image_filenames, test_size=0.2)
    
    #Code to make sure there's one of each class in every dataset
    label_dict_train = []
    label_train = []
    for i in range(len(train_filenames)):
        i = i+1
        basename = os.path.basename(train_filenames[i])
        blocks = basename.split('_')
        if len(blocks) == 6:
            label_train = blocks[0] + '' + blocks[1]
        else:
            label_train = blocks [0]
        if label_train not in label_dict_train:
            label_dict_train.append(label_train)
        if len(label_dict_train) == 51:
            break

    if len(label_dict_train) != 51:
        print('Error! Found only' + str(len(label_dict_train)) + 'Classes for the trainning dataset')
        exit(0)



    label_dict_validation = []
    label_validation = []
    for i in range(len(validation_filenames)):
        i = i+1
        basename = os.path.basename(validation_filenames[i])
        blocks = basename.split('_')
        if len(blocks) == 6:
            label_validation = blocks[0] + '' + blocks[1]
        else:
            label_validation = blocks [0]
        if label_validation not in label_dict_validation:
            label_dict_validation.append(label_validation)
        if len(label_dict_validation) == 51:
            break

    if len(label_dict_validation) != 51:
        print('Error! Found only' + str(len(label_dict_validation)) + 'Classes for the trainning dataset')
        exit(0)
 




    #validation_filenames, test_filenames = train_test_split(remaining_filenames, test_size=0.33)

    print('We have a total of ' + str(len(image_filenames)) + ' images.')
    print('Used ' + str(len(train_filenames)) + ' train images')
    print('Used ' + str(len(validation_filenames)) + ' validation images')
    #print('Used ' + str(len(test_filenames)) + ' test images')

    d = {'train_filenames': train_filenames,
         'validation_filenames': validation_filenames,
         #'test_filenames': test_filenames
         }

    json_object = json.dumps(d, indent=1)

    # Writing to sample.json
    with open("dataset_filenames.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main()
