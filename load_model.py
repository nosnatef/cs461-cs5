import torch
from model.simple import LinearRegression


def main():
    model = LinearRegression()
    model.load_state_dict(torch.load('simple.pt'))
    model.eval()

    prediction = model(12)
    print('prediction: ', prediction)


if __name__ == "__main__":
    main()