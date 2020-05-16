import adsk.core, adsk.fusion, adsk.cam, traceback
import socket 
import json 
import math

#from config import ip_address

# Global list to keep all event handlers in scope.
# This is only needed with Python.
_handlers = []

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface

        # Get the CommandDefinitions collection.
        cmdDefs = ui.commandDefinitions

        # Create a button command definition.
        buttonSample = cmdDefs.itemById('NewButtonDefId')
        if not buttonSample:
            buttonSample = cmdDefs.addButtonDefinition('NewButtonDefId', 'Design time', 'Design time button tooltip', './resources')
        
        # Connect to the command created event.
        onCommandCreated = MyCommandCreatedHandler()
        buttonSample.commandCreated.add(onCommandCreated)
        _handlers.append(onCommandCreated)
        
        # Add a new panel.
        modelWS = ui.workspaces.itemById('FusionSolidEnvironment')

        myPanel = modelWS.toolbarPanels.add('MyPanel', 'Design', 'SolidScriptsAddinsPanel', False)

        # Add the first button to the panel and make it visible in the panel.
        buttonControl = myPanel.controls.addCommand(buttonSample)
        buttonControl.isPromotedByDefault = True
        buttonControl.isPromoted = True
        
        # Execute the command.
        buttonSample.execute()
        
        # Keep the script running.
        adsk.autoTerminate(False)
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))


class MyCommandDestroyHandler(adsk.core.CommandEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            eventArgs = adsk.core.CommandEventArgs.cast(args)

            # when the command is done, terminate the script
            # this will release all globals which will remove all event handlers
            adsk.terminate()
        except:
            if _ui:
                _ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))


# Event handler for the commandCreated event.
class MyCommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        eventArgs = adsk.core.CommandCreatedEventArgs.cast(args)
        
        # Get the command
        cmd = eventArgs.command

        # Get the CommandInputs collection to create new command inputs.            
        inputs = cmd.commandInputs

        numInput1 = inputs.addIntegerSpinnerCommandInput('geometries', 'Number of Geometries', 0 , 1000 , 1, 0)

        numInput2 = inputs.addIntegerSpinnerCommandInput('loads', 'Number of Loads', 0 , 1000 , 1, 0)
        
        numInput3 = inputs.addIntegerSpinnerCommandInput('load_cases', 'Number of load_cases', 0 , 1000 , 1, 0)

        numInput4 = inputs.addIntegerSpinnerCommandInput('keep_ins', 'Number of keep_ins', 0 , 1000 , 1, 0)

        numInput5 = inputs.addIntegerSpinnerCommandInput('keep_outs', 'Number of keep_outs', 0 , 1000 , 1, 0)
        
        numInput6 = inputs.addFloatSpinnerCommandInput('voxels', 'Number of voxels', '', 0 , 100000000, 1, 0)

        # Connect to the execute event.
        onExecute = MyCommandExecuteHandler()
        cmd.execute.add(onExecute)
        _handlers.append(onExecute)

# Event handler for the execute event. 
class MyCommandExecuteHandler(adsk.core.CommandEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        import math
        eventArgs = adsk.core.CommandEventArgs.cast(args)

        app = adsk.core.Application.get()
        ui  = app.userInterface

        # Get the values from the command inputs. 
        inputs = eventArgs.command.commandInputs

        num1 = inputs.itemById('geometries').value

        num2 = inputs.itemById('loads').value

        num3 = inputs.itemById('load_cases').value

        num4 = inputs.itemById('keep_ins').value
        
        num5 = inputs.itemById('keep_outs').value
        
        num6 = inputs.itemById('voxels').value
        
        # Log the voxels
        num7 = round(math.log(num6, 5))

        ui.messageBox('In command execute event handler.')

        # Call the getFeedback function
        getFeedback(num1, num2, num3, num4, num5, num6, num7)


def getFeedback(num1, num2, num3, num4, num5, num6, num7):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface

        # Create a input array
        inputlist = [num1, num2, num3, num4, num5, num6]

        json_str = json.dumps(inputlist)

        ui.messageBox("Starting Connecting!")

        # Connect to the server
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('50.201.138.18', 1234))
        
        # Send the array to server
        s.send(json_str)   

        # Get the feedback for server
        recvbuf = s.recv(1024)

        s.close()

        ui.messageBox("Finish Connecting")
        
        feedback = recvbuf.decode('utf-8')

        ui.messageBox('Feedback:' + feedback)
        
        # Convert string to int
        check = int(feedback)

        # Print the result
        if check < 0:
            ui.messageBox('Invalid message')
        else:
            if feedback == '0':
                ui.messageBox('Short time frame')
        
            if feedback == '1':
                ui.messageBox('Medium time frame')

            if feedback == '2':
                ui.messageBox('Long time frame')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
