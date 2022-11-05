# imports for reading the data sets
import os

# imports for math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

def make_file(is_make):
    if(not is_make):
        return

    # converting data sets to csv
    data_path = "/home/samir/Documents/data/mnist/"
    files = os.listdir(data_path)

    file_list = [None] * 4

    # not needed but ordering the list
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
    fin_train_name = "mnist_train_final.csv"
    fin_test_name = "mnist_test_final.csv"

    data_orig_train.to_csv(data_path+fin_train_name, index=False)
    data_orig_test.to_csv(data_path+fin_test_name, index=False)


if __name__ == "__main__":

    make_file(False)

    # # Reading final files now
    # data_fin_train = pd.read_csv(data_path+fin_train_name)
    # data_fin_test = pd.read_csv(data_path+fin_test_name)


    # # train??
    # ind = 3
    # label_train = data_fin_train['label']
    # print(label_train[ind])

