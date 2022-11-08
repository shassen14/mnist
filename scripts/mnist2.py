# imports for reading the data sets
import os

# imports for math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mnist import MNIST

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

def ReLU(Z):
    return np.maximum(Z,0)

def derivative_ReLU(Z):
    return Z > 0

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
    return exp / exp.sum(axis=0)

def init_params(size):
    W1 = np.random.rand(10,size) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1,b1,W2,b2

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 #10, m
    A1 = ReLU(Z1) # 10,m
    Z2 = W2.dot(A1) + b2 #10,m
    A2 = softmax(Z2) #10,m
    return Z1, A1, Z2, A2

def one_hot(Y):
    ''' return an 0 vector with 1 only in the position correspondind to the value in Y'''
    one_hot_Y = np.zeros((Y.max()+1,Y.size)) #si le chiffre le plus grand dans Y est 9 ca fait 10 lignes
    one_hot_Y[Y,np.arange(Y.size)] = 1 # met un 1 en ligne Y[i] et en colonne i, change l'ordre mais pas le nombre
    return one_hot_Y

def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    one_hot_Y = one_hot(Y)
    dZ2 = 2*(A2 - one_hot_Y) #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2)*derivative_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size

def gradient_descent(X, Y, alpha, iterations):
    size , m = X.shape

    W1, b1, W2, b2 = init_params(size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)

        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)   

        if (i+1) % int(iterations/10) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')
    return W1, b1, W2, b2

############################################################

if __name__ == "__main__":

    data_path = "/home/samir/Documents/data/mnist/"
    fin_train_name = "mnist_train.csv"
    fin_test_name = "mnist_test.csv"

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

    x_train = data_train[1:n_train]/ 255
    y_train = data_train[0]

    mndata = MNIST(data_path)
    images, labels = mndata.load_training()
    x = np.array(images).T / 255
    y = np.array(labels)

    # print(x_train.sum())
    # print(y_train)

    W1, b1, W2, b2 = gradient_descent(x, y, 0.10, 500)




