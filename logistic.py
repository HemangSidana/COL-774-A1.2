import sys
import numpy as np

def write(array,file):
    np.savetxt(file, array, delimiter='\n')

def read_csv(file):
    array = np.loadtxt(file, skiprows=1, delimiter=',')
    return array

def read(file):
    array = np.loadtxt(file)
    return array

if len(sys.argv)>1:
    type= sys.argv[1]
    print(len(sys.argv))
    if type=='a' and len(sys.argv)==5:
        train= read_csv(sys.argv[2])
        parameters = read(sys.argv[3])
        output_model_weight = None
        write(output_model_weight,sys.argv[4])

    elif type=='b' and len(sys.argv)==6:
        train= read_csv(sys.argv[2])
        test = read_csv(sys.argv[3])
        output_model_weight = None
        output_model_pred = None
        write(output_model_weight,sys.argv[4])
        write(output_model_pred,sys.argv[5])
    else:
        print("Wrong Arguments")
else:
    print("No Arguments")