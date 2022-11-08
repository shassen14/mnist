import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mnist import MNIST


data_path = "/home/samir/Documents/data/mnist/"
fin_train_name = "mnist_train_final.csv"
fin_test_name = "mnist_test_final.csv"

mndata = MNIST(data_path)
images, labels = mndata.load_training()

x_train = np.array(images)
y_train = np.array(labels)

print(x_train.shape)
print(y_train.shape)