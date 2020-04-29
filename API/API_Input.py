import adsk.core, adsk.fusion, traceback
import socket

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        
        # Connect to the server
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('3.14.142.108', 1234))

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
            
            # Check that a valid input description was entered.
            unitsMgr = design.unitsManager
            try:
                realValue = unitsMgr.evaluateExpression(input, unitsMgr.defaultLengthUnits)
                isValid = True
            except:
                # Invalid expression so display an error and set the flag to allow them
                # to enter a value again.
                ui.messageBox('"' + input + '" is not a valid input expression.')
                isValid = False

        # Send the input to server
        while True:
            sendbuf = input                
            s.send(sendbuf.encode('utf-8'))   
            if not sendbuf or sendbuf == input:   
                break
        
        # Receive the feedback
        recvbuf = s.recv(1024)
        ui.messageBox("Finish Connecting")
        s.close()
        
        feed = recvbuf.decode('utf-8')
        
        # Check if the input is negative
        check = float(feed)
        if check < 0:
            ui.messageBox('Invalid Input')
        else:
            # Print the result
            ui.messageBox('Feedback:' + feed)

            if feed == '0':
                ui.messageBox('Short time frame')
        
            if feed == '1':
                ui.messageBox('Medium time frame')

            if feed == '2':
                ui.messageBox('Long time frame') 

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))


