from pynput import keyboard, mouse
import logging
from datetime import datetime
import time
import re
import sys
from jsonToTxtConverter import keyboardConverter#, mouseConverter

IDS_PATH = "ids.txt"

## Keyboard Functions
def on_press(key, pressedKeys, file):
    if key == keyboard.Key.esc:
        print("Stoping")
        return False
    
    if key not in pressedKeys:
        pressedKeys.add(key)
        try: 
            print(f'alphanumeric key {str(key.char)} pressed')
            log_dataK(time.time(), 'KeyPressed', str(key.char), file)
        except:
            print(f'special key {str(key)} pressed')
            log_dataK(time.time(), 'SpecialKeyPressed', str(key), file)

def on_release(key, pressedKeys, file):
    if key in pressedKeys:
        print(f"{str(key)} released")
        pressedKeys.discard(key)
        try:
            log_dataK(time.time(), 'KeyReleased', str(key.char), file)
        except AttributeError:
            log_dataK(time.time(), 'SpecialKeyReleased', str(key), file)

## Mouse Dynamics Functions
def on_move(x, y, isClicking, file):
    if isClicking["status"]:
        print('Cursor dragged to {0}'.format((x, y)))
        log_dataM(time.time(), 'MouseDrag', str(x) + ", " + str(y), "null", file)
    else:
        print('Cursor moved to {0}'.format((x, y)))
        log_dataM(time.time(), 'MouseMovement', str(x) + ", " + str(y), "null", file)

def on_scroll(x, y, dx, dy, file):
    print('Mouse scrolled {0} at {1}'.format('down' if dy < 0 else 'up', (x, y)))
    log_dataM(time.time(), 'MouseScroll', str(x) + ", " + str(y), str(dy), file)

def on_click(x, y, button, pressed, isClicking, file):
    isClicking["status"] = pressed
    print('{0} at {1}'.format('Pressed' if pressed else 'Released', (x, y)))
    description = 'MouseClicked' + ('Left' if 'left' in str(button) else 'Right')
    log_dataM(time.time(), description, str(x) + ", " + str(y), "null", file)


## Aux Functions
# Logs
def log_dataK(timestamp, description, key, file):
    log  = "\t\t{\n" \
          f"\t\t\"ts\": {timestamp},\n" \
          f"\t\t\"desc\": \"{description}\",\n" \
          f"\t\t\"key\": \"{key}\"\n" \
           "\t\t},\n"
    file.write(log)
 
def log_dataM(timestamp, description, info, direction, file):
    log  = "\t\t{\n" \
          f"\t\t\"ts\": {timestamp},\n" \
          f"\t\t\"desc\": \"{description}\",\n" \
          f"\t\t\"info\": [{info}],\n" \
          f"\t\t\"dir\": {direction}\n" \
           "\t\t},\n"
    file.write(log)


# User Selection
def invalidUserOption(option, maxNum):
    if option < 0 or option > maxNum:
        return True
    return False

def addUser(IDS_PATH, numUsers):
    with open(IDS_PATH, 'a') as usersFile:
        valid = False

        while not valid:
            userNickname = input(">> Introduce your nickname: ")
            print(f">> Are you sure you are \"{userNickname}\"? (Y/n/q)")
            confirmation = input()
            if re.search(r"(q|quit)", confirmation, flags = re.IGNORECASE):
                print(">> Closing Script ...")
                sys.exit()
            elif re.search(r"(no|n|não|nao)", confirmation, flags = re.IGNORECASE):
                continue
            else:
                valid = True

            usersFile.write(str(userNickname) + ' - ' + str(numUsers + 1) + '\n')
    return numUsers + 1

def selectUser():
    with open(IDS_PATH, 'r') as usersFile:
        print(">> Select your User by the number: \n")
        lines = [line.strip() for line in usersFile.readlines()]
        numUsers = len(lines) - 1
        for line in lines:
            print(line)

        userOption = None
        while userOption is None:
            try:
                userOption = int(input())
                if invalidUserOption(userOption, numUsers):
                    print(f">> {userOption} is not an available Option, Try again")
                    userOption = None
            except ValueError:
                print(">> Error: The inserted value type is not permited")

        userOption = lines[userOption].split(" - ")
        userNickName = userOption[0]
        userId = int(userOption[1])
    if userId == 0:
        userId = addUser(IDS_PATH, numUsers)
    else:
        print(f">> Are you sure you are {userNickName}? (Y/n)")
        confirmation = input()
        if re.search(r"(no|n|não|nao)", confirmation, flags = re.IGNORECASE):
            print(">> Closing Script ...")
            sys.exit()
    return userId

def getTime():
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day

    hour = now.hour
    minute = now.minute
    second = now.second
    return f"{day}-{month}-{year}_{hour}.{minute}.{second}"

def getLogFileName():
    return f"log_id{selectUser()}_{getTime()}.json"

def normalizeJSON(path):
    with open(path, 'r') as file:
        content = file.read()

    with open(path, 'w') as file:
        file.write(content[:-7] + content[-6:])

## Main Function
#TODO: Apply sliding window method
def main():

    logFileName = getLogFileName()

    logKeyboardPath = f'logs/keyboard{logFileName}'
    logMousePath = f'logs/mouse{logFileName}'

    fileKeyboard = open(logKeyboardPath, 'a')
    fileMouse = open(logMousePath, 'a')

    fileKeyboard.write('{\n\t\"record\": [\n')
    fileMouse.write('{\n\t"record\": [\n')

    pressedKeys = set()
    isClicking = {"status": False}

    with keyboard.Listener(on_press = lambda x: on_press(x, pressedKeys, fileKeyboard), on_release = lambda x: on_release(x, pressedKeys, fileKeyboard)) as k_listener, \
        mouse.Listener(on_click=lambda x, y, btn, prs: on_click(x, y, btn, prs, isClicking, fileMouse), on_move = lambda x, y: on_move(x, y, isClicking, fileMouse), on_scroll = lambda x, y, dx, dy: on_scroll(x, y, dx, dy, fileMouse)) as m_listener:
        k_listener.join()
        m_listener.stop()

    fileKeyboard.write('\t]\n}\n')
    fileMouse.write('\t]\n}\n')

    fileKeyboard.close()
    fileMouse.close()

    #Forma mais simples que encontrei de tirar a "," do ultimo caso
    normalizeJSON(logKeyboardPath)
    normalizeJSON(logMousePath)

    keyboardConverter(logKeyboardPath, logFileName[:-4] + "txt")

if __name__ == "__main__":
    main()