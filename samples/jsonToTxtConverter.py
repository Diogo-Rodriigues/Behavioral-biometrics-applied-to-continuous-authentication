import json
import re
from pynput.keyboard import Key

special_keys = list(Key)
key_to_id = {str(k): 1000 + i for i, k in enumerate(special_keys)}

def getKeycode(key):

    print(key)

    try:
        # Caso seja uma tecla comum
        if len(key) == 1:
            return ord(key)
        
        # Para caracteres especiais vamos atribuir o código do pynput mas para não haver conflito de ids com as teclas regulares utilizamos um offset de 1000
        return key_to_id.get(key, 9999) # 9999 para teclas desconhecidas
        
    except Exception:
        return 0

def writeData(PATH_AUTH, fileName, pressEvent, relEvent, COUNTER):
    
    pathToJson = PATH_AUTH + "sessionID.json"
    with open(pathToJson, 'r') as jsonFile:
        data = json.load(jsonFile)

    participantId = str(re.search(r"id(\d)+", fileName).group(1))
    testSectionId = data[participantId]
    sentence = "\"\""
    userInput = "\"\""
    keyStrokeId = str(COUNTER)
    pressTime = int(pressEvent["ts"])
    releaseTime = int(relEvent["ts"])
    letter = str(pressEvent["key"].split(".")[-1])
    
    keyCode = getKeycode(pressEvent['key'])   

    with open(PATH_AUTH + fileName, "a") as file:
        file.write(f"{participantId}\t{testSectionId}\t{sentence}\t{userInput}\t{keyStrokeId}\t{pressTime}\t{releaseTime}\t{letter}\t{keyCode}\n")
    return COUNTER + 1

def keyboardConverter(jsonPath, fileName):

    PATH_AUTH = "../datasets/Keystrokes/AuthorizedUsers/"
    PATH_USER= "../datasets/Keystrokes/AuthorizedUsers/" + fileName
    COUNTER = 1

    with open(PATH_USER, "a") as file:
        file.write('PARTICIPANT_ID\tTEST_SECTION_ID\tSENTENCE\tUSER_INPUT\tKEYSTROKE_ID\tPRESS_TIME\tRELEASE_TIME\tLETTER\tKEYCODE\n')

    with open(jsonPath, 'r') as jsonFile:
        data = json.load(jsonFile)

        record = data["record"]
        while record:
            recPress = record.pop(0)
            pressEvent = {}
            pressEvent["ts"] = recPress["ts"]  
            pressEvent["desc"] = recPress["desc"]
            pressEvent["key"] = recPress["key"]

            idx = None
            for i, value in enumerate(record):
                if value["key"] == pressEvent["key"] and "Released" in value["desc"]:
                    idx = i
                    break

            if idx is not None:
                recRel = record.pop(idx)
                relEvent = {}
                relEvent["ts"]   = recRel["ts"]
                relEvent["desc"] = recRel["desc"]
                relEvent["key"]  = recRel["key"]

                COUNTER = writeData(PATH_AUTH, fileName, pressEvent, relEvent, COUNTER)
    
    PATH2SESSION_JSON = PATH_AUTH + "sessionID.json"
    with open(PATH2SESSION_JSON, 'r') as jsonFile:
        data = json.load(jsonFile)

    data[str(re.search(r"id(\d)+", fileName).group(1))] += 1

    with open(PATH2SESSION_JSON, 'w') as jsonFile:
        json.dump(data, jsonFile, indent=4)
    return            