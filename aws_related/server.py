import socket
import time
import pickle
import json
import numpy as np

# from load_model import *

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 1234))
s.listen(5)

with open ('random_forest_classifier.p', 'rb') as fp:
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
        #decoded_data = float(data.decode('utf-8'))
        data = json.loads(data.decode())
        print("data:", data)
        array = data.get("a")
        print("array:", array)
        print("Array[2]: ", array[2])
        array = np.array(array)
        array[5] = round(math.log10(array[5]), 5)
        array = array.reshape(1, -1)
        print("reshaped array: ", array)
        prediction_data = random_forest_model.predict(array)
        print("model predict:", prediction_data)
        send_back = str(prediction_data[0])
        clientsocket.send(send_back.encode())
    clientsocket.close()
