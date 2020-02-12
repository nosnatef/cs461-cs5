from simple import *

def main():
    model = LinearRegression()
    model.load_state_dict(torch.load('simple.pt'))
    model.eval()

    a = np.array([12])
    a = torch.Tensor(a)
    prediction = model(a)
    prediction = prediction.detach().numpy()
    print('prediction: ', prediction[0])


if __name__ == "__main__":
    main()