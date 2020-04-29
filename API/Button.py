import adsk.core, adsk.fusion, adsk.cam, traceback

_app = adsk.core.Application.cast(None)
_ui = adsk.core.UserInterface.cast(None)

_handlers = []

_standard = adsk.core.DropDownCommandInput.cast(None)


def run(context):
    try:
        global _app, _ui
        _app = adsk.core.Application.get()
        _ui = _app.userInterface
 
        # Get the UserInterface object and the CommandDefinitions collection.
        cmdDefs = _ui.commandDefinitions
		 
        # Create the button command definitions if they don't already exist.
        # Get the existing command definition or create it if it doesn't already exist.
        buttonExample = _ui.commandDefinitions.itemById('NewButtonDefId')
        if not buttonExample:
            buttonExample = cmdDefs.addButtonDefinition('NewButtonDefId', 'Design time', 'Sample button tooltip', './resources')
        
		# Connect to the command created event.
        onCommandCreated = MyCommandCreatedHandler()
        buttonExample.commandCreated.add(onCommandCreated)
        _handlers.append(onCommandCreated)


        # Get the MODEL workspace.
        modelWS = _ui.workspaces.itemById('FusionSolidEnvironment')
        
        # Add a new panel.
        myPanel = modelWS.toolbarPanels.add('myPanel', 'Design', 'SolidScriptsAddinsPanel', False)

        # Add the first button to the panel and make it visible in the panel.
        buttonControl = myPanel.controls.addCommand(buttonExample)
        buttonControl.isPromotedByDefault = True
        buttonControl.isPromoted = True

        buttonExample.execute()
		 
    except:
        if _ui:
            _ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
 


class MyCommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def __init__(self):
        super().__init__()
    def notify(self, args):
        try:

            eventArgs = adsk.core.CommandCreatedEventArgs.cast(args)
            cmd = eventArgs.command

            app = adsk.core.Application.get()
            ui = app.userInterface

            ui.messageBox("Starting!")


        except:
            if _ui:
                _ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))


