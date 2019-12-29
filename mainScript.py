import os

var = int(input("To recognize Face press 1 or to add face press 2 :"))

if var == 1:
    os.system('python recognizer.py')
else:
    os.system('python detector.py')