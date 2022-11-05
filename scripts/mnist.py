# imports for reading the data sets
import os

# imports for math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

############################################################

def convert(img_file, label_file, output_file, n):
    f = open(img_file, "rb")
    l = open(label_file, "rb")
    o = open(output_file, "w")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image  = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    
    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    
    f.close()
    l.close()
    o.close()

############################################################

def make_file(data_path,
              fin_train_name,
              fin_test_name,
              is_make):
    if(not is_make):
        return

    # converting data sets to csv
    files = os.listdir(data_path)

    file_list = [None] * 4

    # not needed but ordering the list, could just use the actual name
    for file in files:
        print('Reading ', file)
        if file.startswith('train-images'):
            file_list[0] = file
        elif file.startswith('train-label'):
            file_list[1] = file
        elif file.startswith('t10k-images'):
            file_list[2] = file
        elif file.startswith('t10k-label'):
            file_list[3] = file

    # Hard coded conversion
    orig_train_name = "mnist_train.csv"
    orig_test_name = "mnist_test.csv"

    print(file_list)

    convert(data_path+file_list[0],
            data_path+file_list[1],
            data_path+orig_train_name,
            60000)

    convert(data_path+file_list[2],
            data_path+file_list[3],
            data_path+orig_test_name,
            10000)

    data_orig_train = pd.read_csv(data_path+orig_train_name)
    data_orig_test = pd.read_csv(data_path+orig_test_name)

    data_orig_train.rename(columns={'5':'label'}, inplace=True)
    data_orig_test.rename(columns={'7':'label'}, inplace=True)

    # Writing final version of files
    data_orig_train.to_csv(data_path+fin_train_name, index=False)
    data_orig_test.to_csv(data_path+fin_test_name, index=False)

############################################################

def init_params(size):
    W1 = np.random.rand(10, size) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

############################################################

def ReLU(Z):
    return np.maximum(0, Z)

############################################################

def ReLU_deriv(Z):
    return Z > 0

############################################################

def soft_max(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

############################################################

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = soft_max(Z2)
    return Z1, A1, Z2, A2

############################################################

def one_hot(Y):
    ''' return a 0 vector with 1 only in the position correspondind to the value in Y'''
    one_hot_y = np.zeros((Y.max()+1 ,Y .size)) 
    one_hot_y[Y, np.arange(Y.size)] = 1 
    return one_hot_y

############################################################

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m):
    one_hot_y = one_hot(Y)
    dZ2 = A2 - one_hot_y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

############################################################

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

############################################################

def get_predictions(A2):
    return np.argmax(A2, 0)

############################################################

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

############################################################

def gradient_descent(X, Y, alpha, iterations):
    size, m = X.shape

    W1, b1, W2, b2 = init_params(size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(f"{get_accuracy(predictions, Y):.3%}")
    return W1, b1, W2, b2

############################################################

if __name__ == "__main__":

    data_path = "/home/samir/Documents/data/mnist/"
    fin_train_name = "mnist_train_final.csv"
    fin_test_name = "mnist_test_final.csv"

    make_file(data_path,
              fin_train_name,
              fin_test_name,
              False)

    # Reading final files now
    data_fin_train = pd.read_csv(data_path+fin_train_name)
    data_fin_test = pd.read_csv(data_path+fin_test_name)

    # convert to numpy array
    data_train = np.array(data_fin_train)
    m_train, n_train = data_train.shape
    np.random.shuffle(data_train)
    data_train = data_train[0:m_train].T

    x_train = data_train[1:n_train]
    y_train = data_train[0]

    print(x_train.shape)
    print(y_train.shape)

    W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.10, 500)




