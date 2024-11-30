import subprocess
from os.path import abspath
from src.utils import print_sucess

filename = 'main.py'

LIMIT = 100
count = 0
while True:
    
    count+=1
    
    PYTHON_PATH = abspath(".env/bin/python3")
    
    p = subprocess.Popen(PYTHON_PATH + " " +filename, shell=True).wait()
    
    if count >= LIMIT:
        print("Limit reached")
        break
    
    if p != 0:        
        continue
    
    else:
        print_sucess("Process finished")
        break
    

    
    