import pickle
import numpy as np
import torch
import torch.nn as nn

from models.linear_regression_model import *
from models.multi_class_model import *

def main():

    ### Load the Dataset ###

    # name = 'number_of_loads_dataset'
    # name = 'number_of_geometries_dataset'
    # name = 'four_dataset'
    name = 'classify_four_normalized_dataset'
    # name = 'test_dataset'
    entry = name + '.p'

    with open ('data/' + entry, 'rb') as fp:
        dataset = pickle.load(fp) 
 
    print(dataset.shape)
    print(dataset)


    ## Split the Dataset into Training and Testing ##
    xTraining, yTraining, xTest, yTest = split_dataset(dataset)

    # print(xTraining.shape)
    # print(yTraining.shape)
    # print(xTest.shape)
    # print(yTest.shape)
    # print(yTraining)



    ## Load the base framework model ##
    # model = LinearRegression()
    model = MultiClass()

    # model.apply(weights_init)

    ## Load the loss function and optimizer ##
    # loss_fn = torch.nn.MSELoss(reduction = 'mean')
    # loss_fn = torch.nn.NLLLoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.MultiLabelSoftMarginLoss()

    # optimzer = torch.optim.SGD(model.parameters(), lr = 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1, weight_decay=1e-4)

    for epoch in range(1000):
        optimizer.zero_grad()
        # model.train()

        # Forward pass
        torch.sigmoid(xTraining).data > 0.5
        y_pred = model(xTraining)
        # print(y_pred)

        # Compute Loss
        loss = loss_fn(y_pred, yTraining)

        # Backward pass
        # optimzer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('epoch {}, loss {}'.format(epoch, loss))  

    ## Output Prediction ##
    torch.sigmoid(xTest).data > 0.5
    y_pred = model(xTest)
    y_pred = torch.argmax(y_pred, 1)
    # y_pred = y_pred.unsqueeze(0)
    # target = torch.zeros(y_pred.size(0), 3).scatter_(1, y_pred, 1.)
    print(y_pred)
    yTest = torch.argmax(yTest, 1)
    print(yTest)

    correct = (y_pred == yTest).sum()
    train_accuracy = 100 * correct / len(y_pred)
    print(train_accuracy)
    # result = (y_pred - yTest) ** 2
    # result = result.detach().numpy()
    # avg_result = np.sum(result) / result.size
    # print("predicted Y value: \n", y_pred)
    # print("test loss: \n",  avg_result)


    torch.save(model.state_dict(), 'models/saved_models/' + name + '.pt')

def weights_init(m):
    torch.nn.init.xavier_uniform(m.weight.data)

def split_dataset(dataset):
    indices = np.random.permutation(dataset.shape[0])    

    idx = round(len(dataset)*0.8)

    xData = len(dataset[0]) - 1
    xData = -3

    training_idx, test_idx = indices[:idx], indices[idx:]
    xTraining, yTraining, xTest, yTest = dataset[training_idx, :xData], dataset[training_idx, xData:], dataset[test_idx, :xData], dataset[test_idx, xData:]

    xTraining = torch.Tensor(xTraining)
    yTraining = torch.Tensor(yTraining)
    xTest     = torch.Tensor(xTest)
    yTest     = torch.Tensor(yTest)

    return xTraining, yTraining, xTest, yTest



if __name__ == "__main__":
    main()