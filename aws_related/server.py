import socket
import time
import pickle
import json

# from load_model import *

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 1234))
s.listen(5)

pickle.loads()
with open ('random_forest_classifier', 'rb') as fp:
    random_forest_model = pickle.load(fp)

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()
    print("Connection from has been established.")
    # clientsocket.send(bytes("Hey there!!!"))
    while True:
        data = clientsocket.recv(1024)
        time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break
        print("Data: %s, Size: %s" % (data.decode('utf-8'), len(data)))
        decoded_data = float(data.decode('utf-8'))
        decoded_data = json.load(decoded_data)
        # prediction_data = output_prediction(decoded_data)
        prediction_data = random_forest_model.predict(decoded_data)
        clientsocket.send(str(prediction_data))
    clientsocket.close()
