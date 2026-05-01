from pynput import keyboard, mouse
import logging
from datetime import datetime
import re
import sys

## Keyboard Functions
def on_press(key, pressedKeys, file):
    if key == keyboard.Key.esc:
        print("Stoping")
        return False
    
    if key not in pressedKeys:
        pressedKeys.add(key)
        try: 
            print(f'alphanumeric key {str(key.char)} pressed')
            log_dataK(datetime.now(), 'KeyPressed', str(key.char), file)
        except:
            print(f'special key {str(key)} pressed')
            log_dataK(datetime.now(), 'SpecialKeyPressed', str(key), file)

def on_release(key, pressedKeys, file):
    if key in pressedKeys:
        print(f"{str(key)} released")
        pressedKeys.discard(key)
    try:
        log_dataK(datetime.now(), 'KeyReleased', str(key.char), file)
    except AttributeError:
        log_dataK(datetime.now(), 'SpecialKeyReleased', str(key), file)

## Mouse Dynamics Functions
def on_move(x, y, file):
    print('Cursor moved to {0}'.format((x, y)))
    log_dataM(datetime.now(), 'MouseMovement', str(x) + ", " + str(y), file)

def on_scroll(x, y, dx, dy, file):
    print('Mouse scrolled {0} at {1}'.format('down' if dy < 0 else 'up', (x, y)))
    log_dataM(datetime.now(), 'MouseScroll', str(x) + ", " + str(y) + "; " + str(dx) + ", " + str(dy), file)

def on_click(x, y, button, pressed, file):
    print('{0} at {1}'.format('Pressed' if pressed else 'Released', (x, y)))
    log_dataM(datetime.now(),'MouseClicked', str(button), file)
    if not pressed:
        #stop listener
        print('Gracefully Stopping!')
        return False

## Aux Functions
# Logs
def log_dataK(timestamp, description, key, file):
    file.write(str(timestamp) + ' | ' + description + ' | ' + key + '\n')

def log_dataM(timestamp, description, info, file):
    file.write(str(timestamp) + ' | ' + description + ' | ' + info + '\n')


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

def selectUser(IDS_PATH):
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

def getLogFileName(IDS_PATH):
    return f"log_id{selectUser(IDS_PATH)}_{getTime()}.txt"

## Main Function
def main():
    # Global Variables
    IDS_PATH = "ids.txt"

    logFileName = getLogFileName(IDS_PATH)
    fileKeyboard = open(f'logs/keyboard{logFileName}', 'a')
    fileMouse = open(f'logs/mouse{logFileName}', 'a')

    pressedKeys = set()

    with keyboard.Listener(on_press = lambda x: on_press(x, pressedKeys, fileKeyboard), on_release = lambda x: on_release(x, pressedKeys, fileKeyboard)) as k_listener, \
         mouse.Listener(on_click=lambda x, y, btn, prs: on_click(x, y, btn, prs, fileMouse), on_move = lambda x, y: on_move(x, y, fileMouse), on_scroll = lambda x, y, dx, dy: on_scroll(x, y, dx, dy, fileMouse)) as m_listener:
        k_listener.join()
        m_listener.join()

    

    fileKeyboard.close()
    fileMouse.close()

if __name__ == "__main__":
    main()