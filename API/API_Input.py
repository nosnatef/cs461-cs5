import adsk.core, adsk.fusion, traceback

_ui  = None
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

        
            tabCmdInput1 = inputs.addTabCommandInput('user input', 'User Input')
            tab1ChildInputs = tabCmdInput1.children

            tab1ChildInputs.addTextBoxCommandInput('writable_textBox', 'Text Box', 'Enter a descirption', 2, False)

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
        app = adsk.core.Application.get()

        global _ui
        _ui = app.userInterface
        
        cmdDef = _ui.commandDefinitions.itemById('Design_time')
        if cmdDef:
            cmdDef.deleteMe()

        cmdDef = _ui.commandDefinitions.addButtonDefinition('Design_time', 'Design_time', 'Design_time')

        onCommandCreated = MyCommandCreatedHandler()

        cmdDef.commandCreated.add(onCommandCreated)

        _handlers.append(onCommandCreated)  

        cmdDef.execute()

        _ui.messageBox("Test")
            
        adsk.autoTerminate(False)

    except:
        if _ui:
            _ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
   
