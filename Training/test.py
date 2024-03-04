#!/usr/bin/env python3


import json
import os
import sys
from sklearn.model_selection import train_test_split
from dataset import Dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from model import Model
import torch.nn.functional as F



def main():

    # -----------------------------------------------------------------
    # Hyperparameters initialization
    # -----------------------------------------------------------------
    batch_size = 500
    n_classes = 51

    # -----------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    with open('../Split_dataset/dataset_filenames.json', 'r') as f:
        # Reading from json file
        dataset_filenames = json.load(f)

    test_filenames = dataset_filenames['test_filenames']
    # test_filenames = test_filenames[0:1000]

    print('Used ' + str(len(test_filenames)) + ' for testing ')

    test_dataset = Dataset(test_filenames)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    # Load the trained model
    checkpoint = torch.load('models/checkpoint.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # we are in testing mode
    for batch_idx, (inputs, labels_gt) in enumerate(test_loader):

        # move tensors to device
        inputs = inputs.to(device)
        labels_gt = labels_gt.to(device)

        # Get predicted labels
        labels_predicted = model.forward(inputs)

    # Getting rid of the device part and transform into numbers
    labels_gt_np = labels_gt.cpu().detach().numpy() 

    # Transform predicted labels into probabilities
    predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()
    # print(labels_gt_np)

    # Take probabilities and find the predict label
    predict_label = [sublist.index(max(sublist)) for sublist in predicted_probabilities]
    # print(predict_label)
    # print(len(predict_label))


    # Creat a classification matrix
    result_matrix = np.zeros((n_classes,n_classes))
    for col, line in zip(predict_label, labels_gt_np):
        result_matrix[line][col] += 1
    
    
    # Saving the matrix to see the results better
    np.savetxt('matrix.txt', result_matrix, fmt='%d')

    # Compute precision and recall
    precision_list = []
    recall_list = []

    for i in range(result_matrix.shape[0]):
        element = result_matrix[i, i]
        TP_FP_sum = np.sum(result_matrix[:, i])
        TP_FN_sum = np.sum(result_matrix[i, :])

        result_precision = element / TP_FP_sum if TP_FP_sum != 0 else 0
        result_recall = element / TP_FN_sum if TP_FN_sum != 0 else 0
        
        precision_list.append(result_precision)
        recall_list.append(result_recall)
 
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1_score = 2 * (precision*recall)/(precision+recall)

    print('Precision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F1 score = ' + str(f1_score))


if __name__ == "__main__":
    main()
