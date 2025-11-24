#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on November 20, 2025, at 09:47
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'solo_project_drm'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Dell\\OneDrive\\Desktop\\PSY310\\SOLO PROJECT\\solo_project_drm_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('key_resp_instr1') is None:
        # initialise key_resp_instr1
        key_resp_instr1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_instr1',
        )
    if deviceManager.getDevice('key_resp_instr2') is None:
        # initialise key_resp_instr2
        key_resp_instr2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_instr2',
        )
    if deviceManager.getDevice('key_resp_math') is None:
        # initialise key_resp_math
        key_resp_math = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_math',
        )
    if deviceManager.getDevice('key_resp_instr3') is None:
        # initialise key_resp_instr3
        key_resp_instr3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_instr3',
        )
    if deviceManager.getDevice('key_resp_test') is None:
        # initialise key_resp_test
        key_resp_test = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_test',
        )
    if deviceManager.getDevice('key_resp_exit') is None:
        # initialise key_resp_exit
        key_resp_exit = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_exit',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='Welcome to my experiment.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "instr" ---
    text_instr = visual.TextStim(win=win, name='text_instr',
        text='Memorize the following words to your best abiliy.\n\nPress the SPACE BAR when ready to begin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_instr1 = keyboard.Keyboard(deviceName='key_resp_instr1')
    
    # --- Initialize components for Routine "lureAnger" ---
    textAnger = visual.TextStim(win=win, name='textAnger',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "lureCold_2" ---
    textCold = visual.TextStim(win=win, name='textCold',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "lureSleep_2" ---
    textSleep = visual.TextStim(win=win, name='textSleep',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "instr2" ---
    text_instr2 = visual.TextStim(win=win, name='text_instr2',
        text='Answer the following math problem by entering the correct answer\n\nPress SPACE BAR when ready to begin',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_instr2 = keyboard.Keyboard(deviceName='key_resp_instr2')
    
    # --- Initialize components for Routine "math" ---
    textMath = visual.TextStim(win=win, name='textMath',
        text='(7 + 6) - 10 =',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_math = keyboard.Keyboard(deviceName='key_resp_math')
    
    # --- Initialize components for Routine "instr3" ---
    text_instr3 = visual.TextStim(win=win, name='text_instr3',
        text="That's Correct!\n\nThe final test will present a new 10-word list with new and old words.\n\nIdentify  the previously presented words by pressing -\n'y' for yes (previously presented)\n'n' for no (not previously presented)\n\nPress SPACE BAR to begin.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_instr3 = keyboard.Keyboard(deviceName='key_resp_instr3')
    
    # --- Initialize components for Routine "test" ---
    textTest = visual.TextStim(win=win, name='textTest',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_test = keyboard.Keyboard(deviceName='key_resp_test')
    
    # --- Initialize components for Routine "blank" ---
    textBlank = visual.TextStim(win=win, name='textBlank',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "thanks" ---
    textThanks = visual.TextStim(win=win, name='textThanks',
        text='Thank you for participating (˶˃ ᵕ ˂˶) .ᐟ.ᐟ\n\npress SPACE BAR to exit',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_exit = keyboard.Keyboard(deviceName='key_resp_exit')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[text_welcome],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome* updates
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_welcome.started')
            # update status
            text_welcome.status = STARTED
            text_welcome.setAutoDraw(True)
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            # update params
            pass
        
        # if text_welcome is stopping this frame...
        if text_welcome.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_welcome.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                text_welcome.tStop = t  # not accounting for scr refresh
                text_welcome.tStopRefresh = tThisFlipGlobal  # on global time
                text_welcome.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_welcome.stopped')
                # update status
                text_welcome.status = FINISHED
                text_welcome.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=welcome,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if welcome.maxDurationReached:
        routineTimer.addTime(-welcome.maxDuration)
    elif welcome.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instr" ---
    # create an object to store info about Routine instr
    instr = data.Routine(
        name='instr',
        components=[text_instr, key_resp_instr1],
    )
    instr.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instr1
    key_resp_instr1.keys = []
    key_resp_instr1.rt = []
    _key_resp_instr1_allKeys = []
    # store start times for instr
    instr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr.tStart = globalClock.getTime(format='float')
    instr.status = STARTED
    thisExp.addData('instr.started', instr.tStart)
    instr.maxDuration = None
    # keep track of which components have finished
    instrComponents = instr.components
    for thisComponent in instr.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr" ---
    instr.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instr* updates
        
        # if text_instr is starting this frame...
        if text_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instr.frameNStart = frameN  # exact frame index
            text_instr.tStart = t  # local t and not account for scr refresh
            text_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instr.started')
            # update status
            text_instr.status = STARTED
            text_instr.setAutoDraw(True)
        
        # if text_instr is active this frame...
        if text_instr.status == STARTED:
            # update params
            pass
        
        # *key_resp_instr1* updates
        waitOnFlip = False
        
        # if key_resp_instr1 is starting this frame...
        if key_resp_instr1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instr1.frameNStart = frameN  # exact frame index
            key_resp_instr1.tStart = t  # local t and not account for scr refresh
            key_resp_instr1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instr1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_instr1.started')
            # update status
            key_resp_instr1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_instr1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_instr1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_instr1.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_instr1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instr1_allKeys.extend(theseKeys)
            if len(_key_resp_instr1_allKeys):
                key_resp_instr1.keys = _key_resp_instr1_allKeys[-1].name  # just the last key pressed
                key_resp_instr1.rt = _key_resp_instr1_allKeys[-1].rt
                key_resp_instr1.duration = _key_resp_instr1_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr" ---
    for thisComponent in instr.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr
    instr.tStop = globalClock.getTime(format='float')
    instr.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr.stopped', instr.tStop)
    # check responses
    if key_resp_instr1.keys in ['', [], None]:  # No response was made
        key_resp_instr1.keys = None
    thisExp.addData('key_resp_instr1.keys',key_resp_instr1.keys)
    if key_resp_instr1.keys != None:  # we had a response
        thisExp.addData('key_resp_instr1.rt', key_resp_instr1.rt)
        thisExp.addData('key_resp_instr1.duration', key_resp_instr1.duration)
    thisExp.nextEntry()
    # the Routine "instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trialsAnger = data.TrialHandler2(
        name='trialsAnger',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('drm_words.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trialsAnger)  # add the loop to the experiment
    thisTrialsAnger = trialsAnger.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrialsAnger.rgb)
    if thisTrialsAnger != None:
        for paramName in thisTrialsAnger:
            globals()[paramName] = thisTrialsAnger[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrialsAnger in trialsAnger:
        trialsAnger.status = STARTED
        if hasattr(thisTrialsAnger, 'status'):
            thisTrialsAnger.status = STARTED
        currentLoop = trialsAnger
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrialsAnger.rgb)
        if thisTrialsAnger != None:
            for paramName in thisTrialsAnger:
                globals()[paramName] = thisTrialsAnger[paramName]
        
        # --- Prepare to start Routine "lureAnger" ---
        # create an object to store info about Routine lureAnger
        lureAnger = data.Routine(
            name='lureAnger',
            components=[textAnger],
        )
        lureAnger.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textAnger.setText(testAnger
        )
        # store start times for lureAnger
        lureAnger.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        lureAnger.tStart = globalClock.getTime(format='float')
        lureAnger.status = STARTED
        thisExp.addData('lureAnger.started', lureAnger.tStart)
        lureAnger.maxDuration = None
        # keep track of which components have finished
        lureAngerComponents = lureAnger.components
        for thisComponent in lureAnger.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "lureAnger" ---
        lureAnger.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsAnger, 'status') and thisTrialsAnger.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textAnger* updates
            
            # if textAnger is starting this frame...
            if textAnger.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textAnger.frameNStart = frameN  # exact frame index
                textAnger.tStart = t  # local t and not account for scr refresh
                textAnger.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textAnger, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textAnger.started')
                # update status
                textAnger.status = STARTED
                textAnger.setAutoDraw(True)
            
            # if textAnger is active this frame...
            if textAnger.status == STARTED:
                # update params
                pass
            
            # if textAnger is stopping this frame...
            if textAnger.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textAnger.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    textAnger.tStop = t  # not accounting for scr refresh
                    textAnger.tStopRefresh = tThisFlipGlobal  # on global time
                    textAnger.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textAnger.stopped')
                    # update status
                    textAnger.status = FINISHED
                    textAnger.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=lureAnger,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                lureAnger.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in lureAnger.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "lureAnger" ---
        for thisComponent in lureAnger.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for lureAnger
        lureAnger.tStop = globalClock.getTime(format='float')
        lureAnger.tStopRefresh = tThisFlipGlobal
        thisExp.addData('lureAnger.stopped', lureAnger.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if lureAnger.maxDurationReached:
            routineTimer.addTime(-lureAnger.maxDuration)
        elif lureAnger.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[textBlank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsAnger, 'status') and thisTrialsAnger.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textBlank* updates
            
            # if textBlank is starting this frame...
            if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlank.frameNStart = frameN  # exact frame index
                textBlank.tStart = t  # local t and not account for scr refresh
                textBlank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.started')
                # update status
                textBlank.status = STARTED
                textBlank.setAutoDraw(True)
            
            # if textBlank is active this frame...
            if textBlank.status == STARTED:
                # update params
                pass
            
            # if textBlank is stopping this frame...
            if textBlank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textBlank.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textBlank.tStop = t  # not accounting for scr refresh
                    textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                    textBlank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textBlank.stopped')
                    # update status
                    textBlank.status = FINISHED
                    textBlank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank.maxDurationReached:
            routineTimer.addTime(-blank.maxDuration)
        elif blank.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        # mark thisTrialsAnger as finished
        if hasattr(thisTrialsAnger, 'status'):
            thisTrialsAnger.status = FINISHED
        # if awaiting a pause, pause now
        if trialsAnger.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trialsAnger.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trialsAnger'
    trialsAnger.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    trialsCold = data.TrialHandler2(
        name='trialsCold',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('lureCold.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trialsCold)  # add the loop to the experiment
    thisTrialsCold = trialsCold.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrialsCold.rgb)
    if thisTrialsCold != None:
        for paramName in thisTrialsCold:
            globals()[paramName] = thisTrialsCold[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrialsCold in trialsCold:
        trialsCold.status = STARTED
        if hasattr(thisTrialsCold, 'status'):
            thisTrialsCold.status = STARTED
        currentLoop = trialsCold
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrialsCold.rgb)
        if thisTrialsCold != None:
            for paramName in thisTrialsCold:
                globals()[paramName] = thisTrialsCold[paramName]
        
        # --- Prepare to start Routine "lureCold_2" ---
        # create an object to store info about Routine lureCold_2
        lureCold_2 = data.Routine(
            name='lureCold_2',
            components=[textCold],
        )
        lureCold_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textCold.setText(lureCold)
        # store start times for lureCold_2
        lureCold_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        lureCold_2.tStart = globalClock.getTime(format='float')
        lureCold_2.status = STARTED
        thisExp.addData('lureCold_2.started', lureCold_2.tStart)
        lureCold_2.maxDuration = None
        # keep track of which components have finished
        lureCold_2Components = lureCold_2.components
        for thisComponent in lureCold_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "lureCold_2" ---
        lureCold_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsCold, 'status') and thisTrialsCold.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textCold* updates
            
            # if textCold is starting this frame...
            if textCold.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textCold.frameNStart = frameN  # exact frame index
                textCold.tStart = t  # local t and not account for scr refresh
                textCold.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textCold, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textCold.started')
                # update status
                textCold.status = STARTED
                textCold.setAutoDraw(True)
            
            # if textCold is active this frame...
            if textCold.status == STARTED:
                # update params
                pass
            
            # if textCold is stopping this frame...
            if textCold.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textCold.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    textCold.tStop = t  # not accounting for scr refresh
                    textCold.tStopRefresh = tThisFlipGlobal  # on global time
                    textCold.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textCold.stopped')
                    # update status
                    textCold.status = FINISHED
                    textCold.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=lureCold_2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                lureCold_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in lureCold_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "lureCold_2" ---
        for thisComponent in lureCold_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for lureCold_2
        lureCold_2.tStop = globalClock.getTime(format='float')
        lureCold_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('lureCold_2.stopped', lureCold_2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if lureCold_2.maxDurationReached:
            routineTimer.addTime(-lureCold_2.maxDuration)
        elif lureCold_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[textBlank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsCold, 'status') and thisTrialsCold.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textBlank* updates
            
            # if textBlank is starting this frame...
            if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlank.frameNStart = frameN  # exact frame index
                textBlank.tStart = t  # local t and not account for scr refresh
                textBlank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.started')
                # update status
                textBlank.status = STARTED
                textBlank.setAutoDraw(True)
            
            # if textBlank is active this frame...
            if textBlank.status == STARTED:
                # update params
                pass
            
            # if textBlank is stopping this frame...
            if textBlank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textBlank.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textBlank.tStop = t  # not accounting for scr refresh
                    textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                    textBlank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textBlank.stopped')
                    # update status
                    textBlank.status = FINISHED
                    textBlank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank.maxDurationReached:
            routineTimer.addTime(-blank.maxDuration)
        elif blank.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        # mark thisTrialsCold as finished
        if hasattr(thisTrialsCold, 'status'):
            thisTrialsCold.status = FINISHED
        # if awaiting a pause, pause now
        if trialsCold.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trialsCold.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trialsCold'
    trialsCold.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    trialsSleep = data.TrialHandler2(
        name='trialsSleep',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('lureSleep.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trialsSleep)  # add the loop to the experiment
    thisTrialsSleep = trialsSleep.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrialsSleep.rgb)
    if thisTrialsSleep != None:
        for paramName in thisTrialsSleep:
            globals()[paramName] = thisTrialsSleep[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrialsSleep in trialsSleep:
        trialsSleep.status = STARTED
        if hasattr(thisTrialsSleep, 'status'):
            thisTrialsSleep.status = STARTED
        currentLoop = trialsSleep
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrialsSleep.rgb)
        if thisTrialsSleep != None:
            for paramName in thisTrialsSleep:
                globals()[paramName] = thisTrialsSleep[paramName]
        
        # --- Prepare to start Routine "lureSleep_2" ---
        # create an object to store info about Routine lureSleep_2
        lureSleep_2 = data.Routine(
            name='lureSleep_2',
            components=[textSleep],
        )
        lureSleep_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textSleep.setText(lureSleep)
        # store start times for lureSleep_2
        lureSleep_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        lureSleep_2.tStart = globalClock.getTime(format='float')
        lureSleep_2.status = STARTED
        thisExp.addData('lureSleep_2.started', lureSleep_2.tStart)
        lureSleep_2.maxDuration = None
        # keep track of which components have finished
        lureSleep_2Components = lureSleep_2.components
        for thisComponent in lureSleep_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "lureSleep_2" ---
        lureSleep_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsSleep, 'status') and thisTrialsSleep.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textSleep* updates
            
            # if textSleep is starting this frame...
            if textSleep.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textSleep.frameNStart = frameN  # exact frame index
                textSleep.tStart = t  # local t and not account for scr refresh
                textSleep.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textSleep, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textSleep.started')
                # update status
                textSleep.status = STARTED
                textSleep.setAutoDraw(True)
            
            # if textSleep is active this frame...
            if textSleep.status == STARTED:
                # update params
                pass
            
            # if textSleep is stopping this frame...
            if textSleep.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textSleep.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    textSleep.tStop = t  # not accounting for scr refresh
                    textSleep.tStopRefresh = tThisFlipGlobal  # on global time
                    textSleep.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textSleep.stopped')
                    # update status
                    textSleep.status = FINISHED
                    textSleep.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=lureSleep_2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                lureSleep_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in lureSleep_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "lureSleep_2" ---
        for thisComponent in lureSleep_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for lureSleep_2
        lureSleep_2.tStop = globalClock.getTime(format='float')
        lureSleep_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('lureSleep_2.stopped', lureSleep_2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if lureSleep_2.maxDurationReached:
            routineTimer.addTime(-lureSleep_2.maxDuration)
        elif lureSleep_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[textBlank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsSleep, 'status') and thisTrialsSleep.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textBlank* updates
            
            # if textBlank is starting this frame...
            if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlank.frameNStart = frameN  # exact frame index
                textBlank.tStart = t  # local t and not account for scr refresh
                textBlank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.started')
                # update status
                textBlank.status = STARTED
                textBlank.setAutoDraw(True)
            
            # if textBlank is active this frame...
            if textBlank.status == STARTED:
                # update params
                pass
            
            # if textBlank is stopping this frame...
            if textBlank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textBlank.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textBlank.tStop = t  # not accounting for scr refresh
                    textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                    textBlank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textBlank.stopped')
                    # update status
                    textBlank.status = FINISHED
                    textBlank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank.maxDurationReached:
            routineTimer.addTime(-blank.maxDuration)
        elif blank.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        # mark thisTrialsSleep as finished
        if hasattr(thisTrialsSleep, 'status'):
            thisTrialsSleep.status = FINISHED
        # if awaiting a pause, pause now
        if trialsSleep.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trialsSleep.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trialsSleep'
    trialsSleep.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instr2" ---
    # create an object to store info about Routine instr2
    instr2 = data.Routine(
        name='instr2',
        components=[text_instr2, key_resp_instr2],
    )
    instr2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instr2
    key_resp_instr2.keys = []
    key_resp_instr2.rt = []
    _key_resp_instr2_allKeys = []
    # store start times for instr2
    instr2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr2.tStart = globalClock.getTime(format='float')
    instr2.status = STARTED
    thisExp.addData('instr2.started', instr2.tStart)
    instr2.maxDuration = None
    # keep track of which components have finished
    instr2Components = instr2.components
    for thisComponent in instr2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr2" ---
    instr2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instr2* updates
        
        # if text_instr2 is starting this frame...
        if text_instr2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instr2.frameNStart = frameN  # exact frame index
            text_instr2.tStart = t  # local t and not account for scr refresh
            text_instr2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instr2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instr2.started')
            # update status
            text_instr2.status = STARTED
            text_instr2.setAutoDraw(True)
        
        # if text_instr2 is active this frame...
        if text_instr2.status == STARTED:
            # update params
            pass
        
        # *key_resp_instr2* updates
        waitOnFlip = False
        
        # if key_resp_instr2 is starting this frame...
        if key_resp_instr2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instr2.frameNStart = frameN  # exact frame index
            key_resp_instr2.tStart = t  # local t and not account for scr refresh
            key_resp_instr2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instr2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_instr2.started')
            # update status
            key_resp_instr2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_instr2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_instr2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_instr2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_instr2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instr2_allKeys.extend(theseKeys)
            if len(_key_resp_instr2_allKeys):
                key_resp_instr2.keys = _key_resp_instr2_allKeys[-1].name  # just the last key pressed
                key_resp_instr2.rt = _key_resp_instr2_allKeys[-1].rt
                key_resp_instr2.duration = _key_resp_instr2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr2" ---
    for thisComponent in instr2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr2
    instr2.tStop = globalClock.getTime(format='float')
    instr2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr2.stopped', instr2.tStop)
    # check responses
    if key_resp_instr2.keys in ['', [], None]:  # No response was made
        key_resp_instr2.keys = None
    thisExp.addData('key_resp_instr2.keys',key_resp_instr2.keys)
    if key_resp_instr2.keys != None:  # we had a response
        thisExp.addData('key_resp_instr2.rt', key_resp_instr2.rt)
        thisExp.addData('key_resp_instr2.duration', key_resp_instr2.duration)
    thisExp.nextEntry()
    # the Routine "instr2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "math" ---
    # create an object to store info about Routine math
    math = data.Routine(
        name='math',
        components=[textMath, key_resp_math],
    )
    math.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_math
    key_resp_math.keys = []
    key_resp_math.rt = []
    _key_resp_math_allKeys = []
    # store start times for math
    math.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    math.tStart = globalClock.getTime(format='float')
    math.status = STARTED
    thisExp.addData('math.started', math.tStart)
    math.maxDuration = None
    # keep track of which components have finished
    mathComponents = math.components
    for thisComponent in math.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "math" ---
    math.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textMath* updates
        
        # if textMath is starting this frame...
        if textMath.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textMath.frameNStart = frameN  # exact frame index
            textMath.tStart = t  # local t and not account for scr refresh
            textMath.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textMath, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textMath.started')
            # update status
            textMath.status = STARTED
            textMath.setAutoDraw(True)
        
        # if textMath is active this frame...
        if textMath.status == STARTED:
            # update params
            pass
        
        # *key_resp_math* updates
        waitOnFlip = False
        
        # if key_resp_math is starting this frame...
        if key_resp_math.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_math.frameNStart = frameN  # exact frame index
            key_resp_math.tStart = t  # local t and not account for scr refresh
            key_resp_math.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_math, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_math.started')
            # update status
            key_resp_math.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_math.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_math.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_math.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_math.getKeys(keyList=['3'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_math_allKeys.extend(theseKeys)
            if len(_key_resp_math_allKeys):
                key_resp_math.keys = _key_resp_math_allKeys[-1].name  # just the last key pressed
                key_resp_math.rt = _key_resp_math_allKeys[-1].rt
                key_resp_math.duration = _key_resp_math_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=math,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            math.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in math.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "math" ---
    for thisComponent in math.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for math
    math.tStop = globalClock.getTime(format='float')
    math.tStopRefresh = tThisFlipGlobal
    thisExp.addData('math.stopped', math.tStop)
    # check responses
    if key_resp_math.keys in ['', [], None]:  # No response was made
        key_resp_math.keys = None
    thisExp.addData('key_resp_math.keys',key_resp_math.keys)
    if key_resp_math.keys != None:  # we had a response
        thisExp.addData('key_resp_math.rt', key_resp_math.rt)
        thisExp.addData('key_resp_math.duration', key_resp_math.duration)
    thisExp.nextEntry()
    # the Routine "math" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr3" ---
    # create an object to store info about Routine instr3
    instr3 = data.Routine(
        name='instr3',
        components=[text_instr3, key_resp_instr3],
    )
    instr3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instr3
    key_resp_instr3.keys = []
    key_resp_instr3.rt = []
    _key_resp_instr3_allKeys = []
    # store start times for instr3
    instr3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr3.tStart = globalClock.getTime(format='float')
    instr3.status = STARTED
    thisExp.addData('instr3.started', instr3.tStart)
    instr3.maxDuration = None
    # keep track of which components have finished
    instr3Components = instr3.components
    for thisComponent in instr3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr3" ---
    instr3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instr3* updates
        
        # if text_instr3 is starting this frame...
        if text_instr3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instr3.frameNStart = frameN  # exact frame index
            text_instr3.tStart = t  # local t and not account for scr refresh
            text_instr3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instr3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instr3.started')
            # update status
            text_instr3.status = STARTED
            text_instr3.setAutoDraw(True)
        
        # if text_instr3 is active this frame...
        if text_instr3.status == STARTED:
            # update params
            pass
        
        # *key_resp_instr3* updates
        waitOnFlip = False
        
        # if key_resp_instr3 is starting this frame...
        if key_resp_instr3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instr3.frameNStart = frameN  # exact frame index
            key_resp_instr3.tStart = t  # local t and not account for scr refresh
            key_resp_instr3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instr3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_instr3.started')
            # update status
            key_resp_instr3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_instr3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_instr3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_instr3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_instr3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instr3_allKeys.extend(theseKeys)
            if len(_key_resp_instr3_allKeys):
                key_resp_instr3.keys = _key_resp_instr3_allKeys[-1].name  # just the last key pressed
                key_resp_instr3.rt = _key_resp_instr3_allKeys[-1].rt
                key_resp_instr3.duration = _key_resp_instr3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr3,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr3" ---
    for thisComponent in instr3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr3
    instr3.tStop = globalClock.getTime(format='float')
    instr3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr3.stopped', instr3.tStop)
    # check responses
    if key_resp_instr3.keys in ['', [], None]:  # No response was made
        key_resp_instr3.keys = None
    thisExp.addData('key_resp_instr3.keys',key_resp_instr3.keys)
    if key_resp_instr3.keys != None:  # we had a response
        thisExp.addData('key_resp_instr3.rt', key_resp_instr3.rt)
        thisExp.addData('key_resp_instr3.duration', key_resp_instr3.duration)
    thisExp.nextEntry()
    # the Routine "instr3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trialsTest = data.TrialHandler2(
        name='trialsTest',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('test_file.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trialsTest)  # add the loop to the experiment
    thisTrialsTest = trialsTest.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrialsTest.rgb)
    if thisTrialsTest != None:
        for paramName in thisTrialsTest:
            globals()[paramName] = thisTrialsTest[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrialsTest in trialsTest:
        trialsTest.status = STARTED
        if hasattr(thisTrialsTest, 'status'):
            thisTrialsTest.status = STARTED
        currentLoop = trialsTest
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrialsTest.rgb)
        if thisTrialsTest != None:
            for paramName in thisTrialsTest:
                globals()[paramName] = thisTrialsTest[paramName]
        
        # --- Prepare to start Routine "test" ---
        # create an object to store info about Routine test
        test = data.Routine(
            name='test',
            components=[textTest, key_resp_test],
        )
        test.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textTest.setText(final_test)
        # create starting attributes for key_resp_test
        key_resp_test.keys = []
        key_resp_test.rt = []
        _key_resp_test_allKeys = []
        # store start times for test
        test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        test.tStart = globalClock.getTime(format='float')
        test.status = STARTED
        thisExp.addData('test.started', test.tStart)
        test.maxDuration = None
        # keep track of which components have finished
        testComponents = test.components
        for thisComponent in test.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test" ---
        test.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsTest, 'status') and thisTrialsTest.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textTest* updates
            
            # if textTest is starting this frame...
            if textTest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textTest.frameNStart = frameN  # exact frame index
                textTest.tStart = t  # local t and not account for scr refresh
                textTest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textTest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textTest.started')
                # update status
                textTest.status = STARTED
                textTest.setAutoDraw(True)
            
            # if textTest is active this frame...
            if textTest.status == STARTED:
                # update params
                pass
            
            # *key_resp_test* updates
            waitOnFlip = False
            
            # if key_resp_test is starting this frame...
            if key_resp_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_test.frameNStart = frameN  # exact frame index
                key_resp_test.tStart = t  # local t and not account for scr refresh
                key_resp_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_test.started')
                # update status
                key_resp_test.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_test.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_test.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_test.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_test.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_test_allKeys.extend(theseKeys)
                if len(_key_resp_test_allKeys):
                    key_resp_test.keys = _key_resp_test_allKeys[-1].name  # just the last key pressed
                    key_resp_test.rt = _key_resp_test_allKeys[-1].rt
                    key_resp_test.duration = _key_resp_test_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_test.keys == str(corrAns)) or (key_resp_test.keys == corrAns):
                        key_resp_test.corr = 1
                    else:
                        key_resp_test.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=test,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                test.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in test.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test" ---
        for thisComponent in test.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for test
        test.tStop = globalClock.getTime(format='float')
        test.tStopRefresh = tThisFlipGlobal
        thisExp.addData('test.stopped', test.tStop)
        # check responses
        if key_resp_test.keys in ['', [], None]:  # No response was made
            key_resp_test.keys = None
            # was no response the correct answer?!
            if str(corrAns).lower() == 'none':
               key_resp_test.corr = 1;  # correct non-response
            else:
               key_resp_test.corr = 0;  # failed to respond (incorrectly)
        # store data for trialsTest (TrialHandler)
        trialsTest.addData('key_resp_test.keys',key_resp_test.keys)
        trialsTest.addData('key_resp_test.corr', key_resp_test.corr)
        if key_resp_test.keys != None:  # we had a response
            trialsTest.addData('key_resp_test.rt', key_resp_test.rt)
            trialsTest.addData('key_resp_test.duration', key_resp_test.duration)
        # the Routine "test" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[textBlank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsTest, 'status') and thisTrialsTest.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textBlank* updates
            
            # if textBlank is starting this frame...
            if textBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textBlank.frameNStart = frameN  # exact frame index
                textBlank.tStart = t  # local t and not account for scr refresh
                textBlank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textBlank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textBlank.started')
                # update status
                textBlank.status = STARTED
                textBlank.setAutoDraw(True)
            
            # if textBlank is active this frame...
            if textBlank.status == STARTED:
                # update params
                pass
            
            # if textBlank is stopping this frame...
            if textBlank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textBlank.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textBlank.tStop = t  # not accounting for scr refresh
                    textBlank.tStopRefresh = tThisFlipGlobal  # on global time
                    textBlank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textBlank.stopped')
                    # update status
                    textBlank.status = FINISHED
                    textBlank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank.maxDurationReached:
            routineTimer.addTime(-blank.maxDuration)
        elif blank.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        # mark thisTrialsTest as finished
        if hasattr(thisTrialsTest, 'status'):
            thisTrialsTest.status = FINISHED
        # if awaiting a pause, pause now
        if trialsTest.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trialsTest.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trialsTest'
    trialsTest.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "thanks" ---
    # create an object to store info about Routine thanks
    thanks = data.Routine(
        name='thanks',
        components=[textThanks, key_resp_exit],
    )
    thanks.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_exit
    key_resp_exit.keys = []
    key_resp_exit.rt = []
    _key_resp_exit_allKeys = []
    # store start times for thanks
    thanks.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thanks.tStart = globalClock.getTime(format='float')
    thanks.status = STARTED
    thisExp.addData('thanks.started', thanks.tStart)
    thanks.maxDuration = None
    # keep track of which components have finished
    thanksComponents = thanks.components
    for thisComponent in thanks.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thanks" ---
    thanks.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textThanks* updates
        
        # if textThanks is starting this frame...
        if textThanks.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textThanks.frameNStart = frameN  # exact frame index
            textThanks.tStart = t  # local t and not account for scr refresh
            textThanks.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textThanks, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textThanks.started')
            # update status
            textThanks.status = STARTED
            textThanks.setAutoDraw(True)
        
        # if textThanks is active this frame...
        if textThanks.status == STARTED:
            # update params
            pass
        
        # *key_resp_exit* updates
        waitOnFlip = False
        
        # if key_resp_exit is starting this frame...
        if key_resp_exit.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_exit.frameNStart = frameN  # exact frame index
            key_resp_exit.tStart = t  # local t and not account for scr refresh
            key_resp_exit.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_exit, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_exit.started')
            # update status
            key_resp_exit.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_exit.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_exit.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_exit.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_exit.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_exit_allKeys.extend(theseKeys)
            if len(_key_resp_exit_allKeys):
                key_resp_exit.keys = _key_resp_exit_allKeys[-1].name  # just the last key pressed
                key_resp_exit.rt = _key_resp_exit_allKeys[-1].rt
                key_resp_exit.duration = _key_resp_exit_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=thanks,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thanks.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thanks.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thanks" ---
    for thisComponent in thanks.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thanks
    thanks.tStop = globalClock.getTime(format='float')
    thanks.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thanks.stopped', thanks.tStop)
    # check responses
    if key_resp_exit.keys in ['', [], None]:  # No response was made
        key_resp_exit.keys = None
    thisExp.addData('key_resp_exit.keys',key_resp_exit.keys)
    if key_resp_exit.keys != None:  # we had a response
        thisExp.addData('key_resp_exit.rt', key_resp_exit.rt)
        thisExp.addData('key_resp_exit.duration', key_resp_exit.duration)
    thisExp.nextEntry()
    # the Routine "thanks" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
