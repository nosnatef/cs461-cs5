from simple import *

def output_prediction(number):
    model = LinearRegression()
    model.load_state_dict(torch.load('simple.pt'))
    model.eval()

    a = np.array([number])
    a = torch.Tensor(a)
    prediction = model(a)
    prediction = prediction.detach().numpy()
    print('prediction: ', prediction[0])
    return prediction[0]
