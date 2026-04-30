# Behavioral-biometrics-applied-to-continuous-authentication
This work implements the application of behavioral biometrics to continuous authentication, focusing on sensing keyboard and mouse interactions as a non-intrusive alternative to static authentication mechanisms.


NOTE:
TO DOWNLOAD THE DATASET FOR THE TRAIN FOLLOW THESE INSTRUCTIONS:

1. Donwload both datasets where:
Mouse Dynamics: https://github.com/balabit/Mouse-Dynamics-Challenge/tree/master
Keystrokes: https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip

2. Create in the root of the project a directory called 'dataset'

3. Inside the 'dataset' directory create another 2 directorys, one called 'Mouse-Dynamics' and other called 'keystroke'

4. Extract the zip files of the datasets directly to each one of those directoies.

The file sctructure should be this:

├── README.md
└── dataset/
    ├── Mouse-Dynamics/
    │   ├── test_files/
    │   │   ├── user7/
    │   │   │   ├── session_0061629194/
    │   │   │   └── ...
    │   │   └── user23/
    │   │       ├── session_0071280153/
    │   │       └── ...
    │   ├── training_files/
    │   │   ├── user7/
    │   │   │   ├── session_1060325796/
    │   │   │   └── ...
    │   │   └── user23/
    │   │       ├── session_0405064924/
    │   │       └── ...
    │   ├── public_labels.txt
    │   └── README.md
    └── keystroke/
        ├── 5_keystrokes.txt
        └── 23_keystrokes.txt
