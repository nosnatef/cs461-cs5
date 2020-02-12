from simple import *

def main():

    with open ('data/dataset.p', 'rb') as fp:
        dataset = pickle.load(fp) 
 
    print(dataset.shape)
    print(dataset)

    indices = np.random.permutation(dataset.shape[0])    

    idx = round(len(dataset)*0.8)

    training_idx, test_idx = indices[:idx], indices[idx:]
    xTraining, yTraining, xTest, yTest = dataset[training_idx, 0], dataset[training_idx, 1], dataset[test_idx, 0], dataset[test_idx, 1]

    xTraining = xTraining.reshape(-1, 1)
    yTraining = yTraining.reshape(-1, 1)
    xTest     = xTest.reshape(-1, 1)
    yTest     = yTest.reshape(-1, 1)

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

    for epoch in range(500):
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
    print("predicted Y value: \n", y_pred)
    print("test loss: \n", (y_pred - yTest) ** 2)

    torch.save(model.state_dict(), 'model/saved_models/simple.pt')




if __name__ == "__main__":
    main()