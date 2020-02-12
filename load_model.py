import torch
import numpy as np
from model.simple import LinearRegression


def main():
    model = LinearRegression()
    model.load_state_dict(torch.load('model/saved_models/simple.pt'))
    model.eval()

    a = np.array([12])
    a = torch.Tensor(a)
    prediction = model(a)
    prediction = prediction.detach().numpy()
    print('prediction: ', prediction[0])


if __name__ == "__main__":
    main()