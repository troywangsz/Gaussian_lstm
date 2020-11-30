from datetime import datetime

import matplotlib.pyplot as plt
from random import seed

import numpy as np
import torch
import pandas as pd
import time
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from cropDataSet import *
from model import *
import settings


def save_model(filename, model):
    state = {'model': model}
    torch.save(state, filename)

def load_model(filename):
    return torch.load(filename)['model']


def RMSE(label, m):
    return float(torch.sqrt(torch.mean(torch.pow(label - m, 2))))


if __name__ == '__main__':

    rand_seed = int(time.time())
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    seed(rand_seed)


    data_all = pd.read_csv("data/1043_199_7_1_sug.csv")
    data_all = data_all.values
    print(data_all.shape)
    data_all = data_all.reshape((-1, 199, 8))
    data_all = torch.tensor(data_all).type(torch.FloatTensor)
    print(data_all.shape)

    dataset = cropDataSet(data_all)

    loader = DataLoader(
        dataset=dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    model = GaussianLstm()
    if settings.USE_CUDA:
        model = model.cuda()
    model.train()

    rmse_valid_low = 100

    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(settings.EPOCHS):
        for i, (sample, label) in enumerate(loader):
            sample = Variable(sample)
            label = Variable(label)
            if settings.USE_CUDA:
                sample = sample.cuda()
                label = label.cuda()

            m, a = model(sample)

            loss = model.loss(label, m, a)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rmse_valid = RMSE(label, m)
            if rmse_valid < rmse_valid_low:
                rmse_valid_low = rmse_valid
                save_model('models/{}-{}-{}-{:.5f}'.format(rand_seed, epoch, i, rmse_valid), model)
                print('lowest rmse valid', rmse_valid)
            print('rmse valid', rmse_valid)

            print('epoch {} batch {}/{} loss: {}'.format(epoch, i, len(loader), loss))