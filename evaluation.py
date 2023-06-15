#!/bin/python

import config
import model
import numpy
import os
import soundfile
import torch
import utils

maximum, minimum = numpy.load(
    os.path.join(config.sourceDir, 'input.bound.npy'),
)

print(f'Maximum: {maximum}')
print(f'Minimum: {minimum}')

rehydrate = utils.rebuilder(maximum, minimum)

audioMood = model.AudioMood()

print('Loading weights from snapshot...')
audioMood.load_state_dict(torch.load(config.modelPath))

evalInputs = model.AudioDataset('evaluation')
evalLoader = torch.utils.data.DataLoader(
    evalInputs,
    batch_size = config.batchSize,
    shuffle = True,
)

audioMood.eval()

result = utils.evaluateSample(audioMood, evalLoader)
target, output = result
lossMaker = torch.nn.MSELoss()
loss = lossMaker(
   torch.from_numpy(output),
   torch.from_numpy(target),
).item()
print(f'  Loss: {loss}')

sample = rehydrate(result)

utils.contrastSample(sample)

length = len(sample)
points = max([len(value) for value in sample])
labels = ['target', 'output']
for y in range(points):
    for x in range(length):
        xLabel = labels[x]
        yLabel = config.dataMoods[:][y]
        print(f'Converting audio {yLabel}.{xLabel}...')
        signal = utils.invertogram(sample[x][y])
        print(f'Saving audio {yLabel}.{xLabel}...')
        soundfile.write(
            os.path.join(config.targetDir, f'audio-{yLabel}-{xLabel}.wav'),
            signal,
            config.audioRate,
            'PCM_16',
        )