import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import *
from cropDataSet import *

def save_model(filename, model):
    state = {'model': model}
    torch.save(state, filename)

def load_model(filename):
    return torch.load(filename)['model']


if __name__ == '__main__':
    model = GaussianLstm()
    model = load_model("models/75-139-0.00")
    model = model.eval()

    data_all = pd.read_csv("data/1043_199_7_1_sug.csv")
    data_all = data_all.values
    print(data_all.shape)
    data_all = data_all.reshape((-1, 199, 8))
    data_all = torch.tensor(data_all).type(torch.FloatTensor)
    print(data_all.shape)

    dataset = cropDataSet(data_all)

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True
    )

    select = 100

    for i in range(select,select+1):
        sample,label = dataset[i:i+1]
        sample = Variable(sample)
        m,a = model(sample)
        pre = model.sample(m,a)
        pre = pre.detach().numpy()
        plt.plot(label.reshape(-1))
        plt.plot(pre.reshape(-1))

    plt.show()