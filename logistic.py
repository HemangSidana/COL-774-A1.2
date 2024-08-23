import sys
import numpy as np
from scipy.special import softmax

def generate_data(data):
    X = data[:, :-1]
    one = np.ones((X.shape[0], 1))
    X= np.hstack((one,X))
    X= X.astype(np.float64)
    Y = data[:, -1]
    Y = Y.astype(int)-1
    unique, counts = np.unique(Y, return_counts=True)
    W =np.zeros((X.shape[1], len(unique)), dtype=np.float64)
    return X,Y,W,counts

def g(W,x,j):
    z = np.dot(x,W)
    sm= softmax(z)
    return sm[j]

def loss(X,Y,W,counts):
    n = X.shape[0]
    Z = X @ W
    softmax_probs = softmax(Z, axis=1)
    indices = (np.arange(n), Y)
    correct_class_probs = softmax_probs[indices]
    scaled_probs = np.log(correct_class_probs) / counts[Y]
    loss_value = -np.mean(scaled_probs) / 2
    return loss_value

def compute_gradient(X, Y, W, counts):
    n, m = X.shape
    k = W.shape[1]
    z = X @ W  
    softmax_probs = softmax(z, axis=1) 
    indices = (np.arange(n), Y)
    Y_one_hot = np.zeros((n, k))
    Y_one_hot[indices] = 1
    grad_W = X.T @ ((softmax_probs - Y_one_hot) / counts[Y][:, np.newaxis]) / (2 * n) 
    return grad_W

def gradient_descent1(X, Y, W, counts, learning_rate, epochs, batch_size):
    n = X.shape[0]
    for i in range(epochs):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X[start:end]
            Y_batch = Y[start:end]
            batch_loss = loss(X_batch, Y_batch, W, counts)
            print(f"Epoch{i+1}, Batch{1+int(start/batch_size)}, Loss{batch_loss}")

            gradient = compute_gradient(X_batch, Y_batch, W, counts)
            W -= learning_rate * gradient

def gradient_descent2(X, Y, W, counts, learning_rate, k, epochs, batch_size):
    n = X.shape[0]
    for i in range(epochs):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X[start:end]
            Y_batch = Y[start:end]
            batch_loss = loss(X_batch, Y_batch, W, counts)
            print(f"Epoch{i+1}, Batch{1+int(start/batch_size)}, Loss{batch_loss}")

            gradient = compute_gradient(X_batch, Y_batch, W, counts)
            W -= learning_rate * gradient /(2+i*k)

def compute_n(X, Y, W, gradient, n0, counts):
    nl = 0.0
    nh = n0
    while loss(X, Y, W, counts) > loss(X, Y, W - nh*gradient, counts):
        nh *= 2
    for _ in range(20):
        n1 = (2*nl + nh)/3
        n2 = (nl + 2*nh)/3
        if loss(X, Y, W - n1*gradient, counts) > loss(X, Y, W - n2*gradient, counts):
            nl = n1
        else:
            nh = n2
    return (nl+nh)/2

def gradient_descent3(X, Y, W, counts, n0, epochs, batch_size):
    print(n0,epochs,batch_size)
    n = X.shape[0]
    for i in range(epochs):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X[start:end]
            Y_batch = Y[start:end]
            batch_loss = loss(X_batch, Y_batch, W, counts)
            print(f"Epoch{i+1}, Batch{1+int(start/batch_size)}, Loss{batch_loss}")

            gradient = compute_gradient(X_batch, Y_batch, W, counts)
            learning_rate= compute_n(X_batch, Y_batch, W, gradient, n0, counts)
            W -= learning_rate * gradient


def compute_n_partb(X, Y, W, gradient, n0, counts):
    nl = 0.0
    nh = n0
    while loss(X, Y, W, counts) > loss(X, Y, W - nh*gradient, counts):
        nh *= 2
    while loss(X, Y, W, counts) < loss(X, Y, W - nh*gradient, counts):
        nh /= 2
    nh *= 2
    for _ in range(20):
        n1 = (2*nl + nh)/3
        n2 = (nl + 2*nh)/3
        if loss(X, Y, W - n1*gradient, counts) > loss(X, Y, W - n2*gradient, counts):
            nl = n1
        else:
            nh = n2
    return nh-nl,(nl+nh)/2

def gradient_descent3_partb(X, Y, W, counts, n0, epochs, batch_size):
    n = X.shape[0]
    n_with_epochs = [n0]*int(np.ceil(n/batch_size))
    for _ in range(epochs):
        batch_num = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X[start:end]
            Y_batch = Y[start:end]
            batch_loss = loss(X_batch, Y_batch, W, counts)
            # print(f"Epoch{i+1}, Batch{1+int(start/batch_size)}, Loss{batch_loss}")
            gradient = compute_gradient(X_batch, Y_batch, W, counts)
            next_n0,learning_rate= compute_n_partb(X_batch, Y_batch, W, gradient, n_with_epochs[batch_num], counts)
            n_with_epochs[batch_num] = next_n0
            W -= learning_rate * gradient
            batch_num += 1


def write(array,file):
    np.savetxt(file, array, delimiter='\n')

def read_csv(file):
    array = np.loadtxt(file, skiprows=1, delimiter=',')
    return array

def read(file):
    array = np.loadtxt(file)
    return array

def read_parameters(filename):
    values = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if i == 1:  # Second line (index 1)
                # Split by comma if present, convert to float, and store as a list
                values.append([float(x) for x in line.split(',')])
            else:
                # Convert other lines to float and store
                values.append(float(line))
    return values

if len(sys.argv)>1:
    type= sys.argv[1]
    if type=='a' and len(sys.argv)==5:
        train= read_csv(sys.argv[2])
        parameters = read_parameters(sys.argv[3])
        X,Y,W,counts = generate_data(train)
        if (parameters[0]==1):
            gradient_descent1(X,Y,W,counts,parameters[1][0],int(parameters[2]),int(parameters[3]))
        elif (parameters[0]==2):
            gradient_descent2(X,Y,W,counts,parameters[1][0],parameters[1][1],int(parameters[2]),int(parameters[3]))
        else:
            gradient_descent3(X,Y,W,counts,parameters[1][0],int(parameters[2]),int(parameters[3]))
        output_model_weight = W.flatten()
        write(output_model_weight,sys.argv[4])

    elif type=='b' and len(sys.argv)==6:
        train= read_csv(sys.argv[2])
        test = read_csv(sys.argv[3])
        X,Y,W,counts = generate_data(train)
        gradient_descent3_partb(X,Y,W,counts,1e-9,5,100)
        output_model_weight = W.flatten()
        X_test = test
        one = np.ones((X_test.shape[0], 1))
        X_test= np.hstack((one,X_test))
        X_test= X_test.astype(np.float64)
        Z = X_test @ W
        output_model_pred = softmax(Z, axis=1)
        write(output_model_weight,sys.argv[4])
        np.savetxt(sys.argv[5], output_model_pred, delimiter=",")
    else:
        print("Wrong Arguments")
else:
    print("No Arguments")