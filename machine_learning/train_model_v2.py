import pandas as pd
import seaborn as sns
import csv
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
from collections import OrderedDict

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import pickle

# from models.linear_regression_model import *
# from models.multi_class_model import *
from models.multi_class_model_v2 import MultiClassV2

class ClassifierDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

def main():

    ## Load in the dataset:

    # name = 'number_of_loads_dataset'
    # name = 'number_of_geometries_dataset'
    # name = 'four_dataset'
    # name = 'classify_four_normalized_dataset'
    # name = 'classify_four_v2_dataset'
    name = 'combined_dataset'
    # entry = name + '.p'
    entry = name + '.csv'

    # with open ('data/' + entry, 'rb') as fp:
    #     dataset = pickle.load(fp) 
    # df_unbalanced = pd.read_csv("data/classify_four_v2_dataset.csv")
    # df_unbalanced = pd.read_csv("data/combined_dataset.csv")
    df_unbalanced = pd.read_csv("data/combined_v2_dataset.csv")
    # df_unbalanced = pd.read_csv("data/combined_v3_dataset.csv")

 
    # print(dataset.shape)
    # print(dataset)
    print(df_unbalanced.head())
    df_short = df_unbalanced.loc[df_unbalanced['time'] == 0]
    df_med = df_unbalanced.loc[df_unbalanced['time'] == 1]
    df_long = df_unbalanced.loc[df_unbalanced['time'] == 2]
    print("short entries: {}".format(len(df_short)))
    print("medium entries: {}".format(len(df_med)))
    print("long entries: {}".format(len(df_long)))
    print("*****")

    df_short_downsampled = resample(df_short, replace=False, n_samples=102, random_state=333)
    # df_med_upsampled = resample(df_med, replace=True, n_samples=551, random_state=933)
    df_long_upsampled = resample(df_long, replace=True, n_samples=102, random_state=432)

    print("new short entries: {}".format(len(df_short_downsampled)))
    print("new medium entries: {}".format(len(df_med)))
    print("new long entries: {}".format(len(df_long_upsampled)))
    print("*****")

    df = pd.concat([df_short_downsampled, df_med, df_long_upsampled])
    df = df.sample(frac=1).reset_index(drop=True)

    print(df.head())

    ## Split data into relevant training and testing sets
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=30)
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=30)

    # Validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.1, stratify=y_train_val, random_state = 44)
    # xTraining, yTraining, xTest, yTest = split_dataset(dataset)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)


    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_test, y_test = np.array(x_test), np.array(y_test)

    X_Train, Y_Train = np.array(X_Train), np.array(Y_Train)
    X_Test, Y_Test = np.array(X_Test), np.array(Y_Test)

    train_dataset = ClassifierDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    val_dataset   = ClassifierDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
    test_dataset  = ClassifierDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())

    # Create list with all outputs and shuffle it
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_count = [i for i in count_class_distribution(y_train).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)

    print(class_weights)

    class_weights_total = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_total,
        num_samples=len(class_weights_total),
        replacement=True
    )

    EPOCHS = 300
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001

    NUM_FEATURES = len(X.columns)
    NUM_CLASSES = 3


    ## Dataloader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #############

    ## Bring in machine learning framework model
    # model = LinearRegression()
    # model = MultiClass()
    model = MultiClassV2(n_in=NUM_FEATURES, n_out=NUM_CLASSES)

    model.to(device)

    # model.apply(weights_init)

    ## Define loss function and optimizer
    # loss_fn = torch.nn.MSELoss(reduction = 'mean')
    # loss_fn = torch.nn.NLLLoss()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.MultiLabelSoftMarginLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.1, weight_decay=1e-4)

    print(model)

    ## Dictionaries to help keep track of accuracy and loss for each epoch
    # accuracy_stats = {
    #     'train': [],
    #     'val': []
    # }
    # loss_stats = {
    #     'train': [],
    #     'val': []
    # }
    training_info = OrderedDict()
    training_info['epoch'] = 0
    training_info['acc_train'] = []
    training_info['acc_val'] = []
    training_info['loss_train'] = []
    training_info['loss_val'] = []

    ############################################################################################################
    # t = trange(1, EPOCHS+1, desc='ML')
    # # for e in tqdm(range(1, EPOCHS+1)):
    # for e in t:
    #     train_epoch_loss = 0
    #     train_epoch_acc = 0

    #     model.train()
    #     for x_train_batch, y_train_batch in train_loader:
    #         x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)
    #         optimizer.zero_grad()

    #         y_train_pred = model(x_train_batch)

    #         train_loss = loss_fn(y_train_pred, y_train_batch)
    #         train_acc = multi_acc(y_train_pred, y_train_batch)

    #         train_loss.backward()
    #         optimizer.step()

    #         train_epoch_loss += train_loss.item()
    #         train_epoch_acc += train_acc.item()
        
    #     with torch.no_grad():
    #         val_epoch_loss = 0
    #         val_epoch_acc = 0

    #         model.eval()
    #         for x_val_batch, y_val_batch in val_loader:
    #             x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)

    #         y_val_pred = model(x_val_batch)

    #         val_loss = loss_fn(y_val_pred, y_val_batch)
    #         val_acc = multi_acc(y_val_pred, y_val_batch)

    #         val_epoch_loss += val_loss.item()
    #         val_epoch_acc += val_acc.item()

    #         loss_train = round(train_epoch_loss/len(train_loader), 5)
    #         loss_val = round(val_epoch_loss/len(train_loader), 5)
    #         acc_train = round(train_epoch_acc/len(train_loader), 3)
    #         acc_val = round(val_epoch_acc/len(val_loader), 3)
    #         training_info['epoch'] = e
    #         training_info['loss_train'].append(loss_train)
    #         training_info['loss_val'].append(loss_val)
    #         training_info['acc_train'].append(acc_train)
    #         training_info['acc_val'].append(acc_val)

    #         t.set_description("Epochs: {:03} | Train Loss: {:.5f} | Val Loss: {:.5f} | Train Acc: {:.3f} | Val Acc: {:.3f}\n".format(e, 
    #                             loss_train, loss_val, acc_train, acc_val))

    #     # loss_stats['train'].append(train_epoch_loss/len(train_loader))
    #     # loss_stats['val'].append(val_epoch_loss/len(val_loader))
    #     # accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    #     # accuracy_stats['val'].append(val_epoch_acc/len(val_loader))

    # # print("Epoch {:03}: | Train Loss: {:.5f} | Val Loss: {:.5f} | Train Acc: {:.3f} | Val Acc: {:.3f}"
    # # .format(e, train_epoch_loss/len(train_loader), val_epoch_loss/len(train_loader),
    # #         train_epoch_acc/len(train_loader), val_epoch_acc/len(train_loader)))

    # # Test the model
    # y_pred_list = []
    # with torch.no_grad():
    #     model.eval()
    #     for X_batch, _ in test_loader:
    #         X_batch = X_batch.to(device)
    #         y_test_pred = model(X_batch)
    #         y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
    #         _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    #         y_pred_list.append(y_pred_tags.cpu().numpy())

    # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
            
    # print(classification_report(y_test, y_pred_list))
    ######################################################################################################################
    # print("SVC TIME")
    # clf_2 = SVC(kernel='linear', class_weight='balanced', probability=True)
    # clf_2.fit(x_train, y_train)
    # y_svc_pred = clf_2.predict(x_test)
    # print(y_svc_pred)
    # print("Accuracy:", metrics.accuracy_score(y_test, y_svc_pred))

    # print("************************************")

    print("RANDOME FORSET TIME")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_Train, Y_Train)

    feature_imp = pd.Series(clf.feature_importances_,index=df.columns[:-1]).sort_values(ascending=False)
    print("feature importance:")
    print(feature_imp)

    y_forest_pred = clf.predict(X_Test)
    print(y_forest_pred)
    print("Accuracy:", metrics.accuracy_score(Y_Test, y_forest_pred))

    file_name = 'random_forest_classifier.p'
    with open('models/saved_models/' + file_name, 'wb') as fp:
        pickle.dump(clf, fp)


    # prob_forest = clf.predict_proba(x_test)
    # prob_forest = [p[1] for p in prob_forest]
    # print("AUROC: ", )



    # ## Train the model
    # for epoch in range(1000):
    #     optimizer.zero_grad()
    #     # model.train()

    #     ## Forward pass
    #     # torch.sigmoid(xTraining).data > 0.5
    #     y_pred = model(xTraining)

    #     ## Compute Loss
    #     loss = loss_fn(y_pred, yTraining)

    #     ## Backward pass
    #     # optimzer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print('epoch {}, loss {}'.format(epoch, loss))  

    # ## Accuracy
    # # torch.sigmoid(xTest).data > 0.5
    # y_pred = model(xTest)
    # print(y_pred)
    # y_pred = torch.argmax(y_pred, 1)
    # # y_pred = y_pred.unsqueeze(0)
    # # target = torch.zeros(y_pred.size(0), 3).scatter_(1, y_pred, 1.)
    # print(y_pred)
    # yTest = torch.argmax(yTest, 1)
    # print(yTest)

    # correct = (y_pred == yTest).sum()
    # train_accuracy = 100 * correct / len(y_pred)
    # print(train_accuracy)
    # # result = (y_pred - yTest) ** 2
    # # result = result.detach().numpy()
    # # avg_result = np.sum(result) / result.size
    # # print("predicted Y value: \n", y_pred)
    # # print("test loss: \n",  avg_result)


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

def count_class_distribution(data):
    count_dict = {
        "short": 0,
        "med": 0,
        "long": 0,
    }

    for time_entry in data:
        if time_entry == 0:
            count_dict['short'] += 1
        elif time_entry == 1:
            count_dict['med'] += 1
        elif time_entry == 2:
            count_dict['long'] += 1
        else:
            print("ERROR: Check classes")
    
    return count_dict

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc) * 100

    return acc


if __name__ == "__main__":
    main()

