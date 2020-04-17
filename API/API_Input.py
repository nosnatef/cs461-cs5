import adsk.core, adsk.fusion, traceback
import socket

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('18.218.162.183', 1234))

        ui.messageBox("Starting Connecting!")

        design = app.activeProduct
        if not design:
            ui.messageBox('No active Fusion design', 'No Design')
            return

        # Prompt the user for a string and validate it's valid.
        isValid = False


        while not isValid:
            # Get a string from the user.
            retVals = ui.inputBox('Enter a integer', 'Design', '')

            if retVals[1] == True:
                return
            
            input = retVals[0]
            
            # Check that a valid length description was entered.
            unitsMgr = design.unitsManager
            try:
                realValue = unitsMgr.evaluateExpression(input, unitsMgr.defaultLengthUnits)
                isValid = True
            except:
                # Invalid expression so display an error and set the flag to allow them
                # to enter a value again.
                ui.messageBox('"' + input + '" is not a valid length expression.')
                isValid = False


        while True:
            sendbuf = input                
            s.send(sendbuf.encode('utf-8'))   
            if not sendbuf or sendbuf == input:   
                break

        recvbuf = s.recv(1024)
        ui.messageBox("Finish Connecting")
        s.close()

        ui.messageBox(recvbuf.decode('utf-8'))

        ui.messageBox('Stop addin')

    
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))


