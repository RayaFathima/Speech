#!/bin/python

import config
import librosa
import model
import numpy
import os
import torch
import utils

maximum, minimum = numpy.load(
    os.path.join(config.sourceDir, 'input.bound.npy'),
)

print(f'Maximum: {maximum}')
print(f'Minimum: {minimum}')

rehydrate = utils.rebuilder(maximum, minimum)

def setWeights(model):
    classname = model.__class__.__name__
    if classname.find('Linear') != -1:
        length = model.in_features
        offset = 1.0 / numpy.sqrt(length)
        model.weight.data.uniform_(-offset, offset)
        model.bias.data.fill_(0)

audioMood = model.AudioMood()

if (os.path.isfile(config.modelPath)):
    print('Loading weights from snapshot...')
    audioMood.load_state_dict(torch.load(config.modelPath))
else:
    print('Initializing uniform weights...')
    audioMood.apply(setWeights)

trainInputs, testInputs, evalInputs = [
    model.AudioDataset(dataType)
    for dataType in config.dataTypes
]

trainLoader, testLoader, evalLoader = [
    torch.utils.data.DataLoader(
        inputs,
        batch_size = config.batchSize,
        shuffle = True,
    )
    for inputs in [trainInputs, testInputs, evalInputs]
]

learnRate = 0.001
optimizer = torch.optim.Adam(audioMood.parameters(), lr = learnRate)
lossMaker = torch.nn.MSELoss()

print('Training network...')

oldLoss = None
for round in range(config.trainings):
    print(f'Round: {round}')
    audioMood.train()
    for batch, (source, target) in enumerate(trainLoader):
        output = audioMood(source)
        change = lossMaker(output, target)
        print(f'  [{batch}]: {change.item()}')

        optimizer.zero_grad()
        change.backward()
        optimizer.step()

    audioMood.eval()
    target, output = utils.evaluateSample(audioMood, testLoader)
    newLoss = lossMaker(
        torch.from_numpy(output),
        torch.from_numpy(target),
    ).item()

    print(f'  Loss: {newLoss}')

    if oldLoss is None or oldLoss > newLoss:
        oldLoss = newLoss
    print('Saving model...')
    torch.save(audioMood.state_dict(), config.modelPath)

audioMood.eval()
utils.contrastSample(
    rehydrate(utils.evaluateSample(audioMood, evalLoader))
)
