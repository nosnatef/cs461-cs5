#Author-
#Description-

import adsk.core, adsk.fusion, adsk.cam, traceback
import socket

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('3.14.246.112', 1234))

        ui.messageBox("starting")

        while True:
            sendbuf = "exit"                
            s.send(sendbuf.encode('utf-8'))   
            if not sendbuf or sendbuf == 'exit':   
                break
        recvbuf = s.recv(1024)
        ui.messageBox("finished")
        s.close()

        
        ui.messageBox(recvbuf.decode('utf-8'))

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))




def stop(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        ui.messageBox('Stop addin')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
