# adapted from https://github.com/saztorralba/CNNWordReco 

import glob
import math
import pandas as pd
import random
import torch
from tqdm import tqdm

from sigproc import gen_logmel, feat2img

def get_rec_paths(path):
    ## labels and paths in pd frame
    wavfiles = glob.glob(path + '/*.wav')
    speakers = [file.split('/')[-1].split('_')[1] for file in wavfiles]
    #words = [list(args['vocab'].keys())[int(file.split('/')[-1].split('_')[0])] for file in wavfiles]
    labels = [int(file.split('/')[-1].split('_')[0]) for file in wavfiles]
    rec_number = [int(file.split('/')[-1].split('_')[2].split('.')[0]) for file in wavfiles]
    #data = pd.DataFrame({'wavfile':wavfiles,'speaker':speakers,'word':words,'rec_number':rec_number})
    frame = pd.DataFrame({'wavfile':wavfiles,'speaker':speakers,'label':labels,'rec_number':rec_number})

    return frame

# log mels for audio; time scaled by PIL.Image to xsize, 40 nmels
def load_data(data,cv=False,**kwargs):
    # train/test split according to https://github.com/Jakobovski/free-spoken-digit-dataset
    train_data = data.loc[data['rec_number']>=5].reset_index(drop=True)
    test_data = data.loc[data['rec_number']<5].reset_index(drop=True)

    # train/val data
    n_samples = len(train_data)
    dataset = torch.zeros((n_samples,kwargs['ysize'],kwargs['xsize']),dtype=torch.uint8)
    labels = torch.zeros((n_samples),dtype=torch.uint8)
    for i in tqdm(range(n_samples),disable=(kwargs['verbose']<2)):
        path = train_data['wavfile'][i]
        dataset[i,:,:] = torch.from_numpy(feat2img(gen_logmel(path,(kwargs['n_mels'] if 'n_mels' in kwargs else 40),(kwargs['sampling'] if 'sampling' in kwargs else 8000),True),kwargs['ysize'],kwargs['xsize']))
        #labels[i] = kwargs['vocab'][train_data['word'][i]]
        labels[i] = train_data['label'][i]

    if cv == False:
        return dataset, labels

    # random train/val split
    idx = [i for i in range(n_samples)]
    random.shuffle(idx)
    trainset = dataset[idx[0:int(n_samples*(1-kwargs['train_val_percentage']))]]
    trainlabels = labels[idx[0:int(n_samples*(1-kwargs['train_val_percentage']))]]
    validset = dataset[idx[int(n_samples*(1-kwargs['train_val_percentage'])):]]
    validlabels = labels[idx[int(n_samples*(1-kwargs['train_val_percentage'])):]]

    # test data
    n_samples = len(test_data)
    testdataset = torch.zeros((n_samples,kwargs['ysize'],kwargs['xsize']),dtype=torch.uint8)
    testlabels = torch.zeros((n_samples),dtype=torch.uint8)
    for i in tqdm(range(n_samples),disable=(kwargs['verbose']<2)):
        path = test_data['wavfile'][i]
        testdataset[i,:,:] = torch.from_numpy(feat2img(gen_logmel(path,(kwargs['n_mels'] if 'n_mels' in kwargs else 40),(kwargs['sampling'] if 'sampling' in kwargs else 8000),True),kwargs['ysize'],kwargs['xsize']))
        #testlabels[i] = kwargs['vocab'][data['word'][i]]
        testlabels[i] = test_data['label'][i]

    return trainset, validset, trainlabels, validlabels, testdataset, testlabels

#Train the model for an epoch
def train_model(trainset,trainlabels,model,optimizer,criterion,**kwargs):
    trainlen = trainset.shape[0]
    nbatches = math.ceil(trainlen/kwargs['batch_size'])
    if trainlen % kwargs['batch_size'] == 1:
        nbatches -= 1
    total_loss = 0
    total_backs = 0
    with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
        model = model.train()
        for b in range(nbatches):

            #Obtain batch
            X = trainset[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().float()
            X = X.to(kwargs['device'])
            Y = trainlabels[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
            #import pdb; pdb.set_trace()

            #Propagate
            posteriors = model(X)

            #Backpropagate
            loss = criterion(posteriors,Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #Track loss
            if total_backs == 100:
                total_loss = total_loss*0.99+loss.detach().cpu().numpy()
            else:
                total_loss += loss.detach().cpu().numpy()
                total_backs += 1
            pbar.set_description(f'Training epoch. Loss {total_loss/(total_backs+1):.2f}')
            pbar.update()
    return total_loss/(total_backs+1)

#Validate last epoch's model
def validate_model(validset,validlabels,model,**kwargs):
    validlen = validset.shape[0]
    acc = 0
    total = 0
    nbatches = math.ceil(validlen/kwargs['batch_size'])
    with torch.no_grad():
        with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
            model = model.eval()
            for b in range(nbatches):
                #Obtain batch
                X = validset[b*kwargs['batch_size']:min(validlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
                Y = validlabels[b*kwargs['batch_size']:min(validlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
                #Propagate
                posteriors = model(X)
                #Accumulate accuracy
                estimated = torch.argmax(posteriors,dim=1)
                acc += sum((estimated.cpu().numpy() == Y.cpu().numpy()))
                total+=Y.shape[0]
                pbar.set_description(f'Evaluating epoch. Accuracy {100*acc/total:.2f}%')
                pbar.update()
    return 100*acc/total