import socket
  
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('3.135.212.76', 1234))


while True:
    full_msg = ''
    while True:
        msg = s.recv(8)
        if len(msg) <= 0:
            break
        full_msg += msg.decode("utf-8")

    if len(full_msg) > 0:
        print(full_msg)
