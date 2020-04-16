import adsk.core, adsk.fusion, traceback
import socket

_ui  = None
_app = None
_handlers = []


class MyCommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:

            cmd = adsk.core.Command.cast(args.command)
            inputs = cmd.commandInputs
            
            onDestroy = MyCommandDestroyHandler()
            cmd.destroy.add(onDestroy)
            _handlers.append(onDestroy)

        
            tabCmdInput1 = inputs.addTabCommandInput('user_input', 'User Input')
            tab1ChildInputs = tabCmdInput1.children

            strInput = tab1ChildInputs.addStringValueInput('string', 'Text', 'Enter a string')

            returnValue = tab1ChildInputs.addValueInput('value', 'Value', '', adsk.core.ValueInput.createByReal(0.0))


        except:
            _ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

            
class MyCommandDestroyHandler(adsk.core.CommandEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:
            adsk.terminate()
        except:
            _ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

def run(context):
    try:

        global _app, _ui

        _app = adsk.core.Application.get()
        _ui = _app.userInterface
        
        cmdDef = _ui.commandDefinitions.itemById('Design_time')
        if cmdDef:
            cmdDef.deleteMe()

        cmdDef = _ui.commandDefinitions.addButtonDefinition('Design_time', 'Design_time', 'Design_time')

        onCommandCreated = MyCommandCreatedHandler()

        cmdDef.commandCreated.add(onCommandCreated)

        _handlers.append(onCommandCreated)  

        cmdDef.execute()
            
        adsk.autoTerminate(False)

    except:
        if _ui:
            _ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))


def stop(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('18.218.162.183', 1234))

        ui.messageBox("Starting Connecting!")

        while True:
            sendbuf = "exit"                
            s.send(sendbuf.encode('utf-8'))   
            if not sendbuf or sendbuf == 'exit':   
                break
        recvbuf = s.recv(1024)
        ui.messageBox("Finish Connecting")
        s.close()

        ui.messageBox(recvbuf.decode('utf-8'))

        ui.messageBox('Stop addin')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
