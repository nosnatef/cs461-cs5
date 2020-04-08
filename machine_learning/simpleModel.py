from simple import *

def main():

    # name = 'number_of_loads_dataset'
    # name = 'number_of_geometries_dataset'
    # name = 'four_dataset'
    name = 'four_normalized_dataset'
    entry = name + '.p'

    with open ('data/' + entry, 'rb') as fp:
        dataset = pickle.load(fp) 
 
    print(dataset.shape)
    print(dataset)

    indices = np.random.permutation(dataset.shape[0])    

    idx = round(len(dataset)*0.8)

    xData = len(dataset[0]) - 1

    training_idx, test_idx = indices[:idx], indices[idx:]
    xTraining, yTraining, xTest, yTest = dataset[training_idx, :xData], dataset[training_idx, xData], dataset[test_idx, :xData], dataset[test_idx, xData]

    # print(xTraining.shape)
    # xTraining = xTraining.reshape(-1, 1)
    # print(xTraining.shape)
    # yTraining = yTraining.reshape(-1, 1)
    # xTest     = xTest.reshape(-1, 1)
    # yTest     = yTest.reshape(-1, 1)

    xTraining = torch.Tensor(xTraining)
    yTraining = torch.Tensor(yTraining)
    xTest     = torch.Tensor(xTest)
    yTest     = torch.Tensor(yTest)

    print(xTraining.shape)
    print(yTraining.shape)
    print(xTest.shape)
    print(yTest.shape)



    model = LinearRegression()

    loss_fn = torch.nn.MSELoss(reduction = 'mean')

    optimzer = torch.optim.SGD(model.parameters(), lr = 0.001)

    for epoch in range(1000):
        model.train()

        # Forward pass
        y_pred = model(xTraining)

        # Compute Loss
        loss = loss_fn(y_pred, yTraining)

        # Backward pass
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        print('epoch {}, loss {}'.format(epoch, loss))  

    y_pred = model(xTest)
    result = (y_pred - yTest) ** 2
    result = result.detach().numpy()
    avg_result = np.sum(result) / result.size
    print("predicted Y value: \n", y_pred)
    print("test loss: \n",  avg_result)


    torch.save(model.state_dict(), 'model/saved_models/' + name + '.pt')




if __name__ == "__main__":
    main()