#!/bin/python

import config
import librosa
import matplotlib.pyplot as plt
import numpy
import os
import random
import utils

block = 20
fixed = random.randint(0, block)

target = {}
for dataType in config.dataTypes:
    source = numpy.load(
        os.path.join(config.sourceDir, f'cache.{dataType}.npy'),
        mmap_mode = 'r',
    )
    length = len(source)
    target[dataType] = []
    print(f'Converting {dataType} signals to spectrogram...')
    for index in range(length):
        target[dataType].append(
            numpy.stack([
                utils.spectrogram(signal)
                for signal in source[index]
            ]),
        )
        if (index == fixed):
            sample = target[dataType][index]
            points = len(sample)
            for point in range(points):
                plt.subplot(points, 1, point + 1);
                librosa.display.specshow(sample[point])
            plt.show()
        if (index % block == 0):
            print(f'- {index} / {length}')
    target[dataType] = numpy.stack(target[dataType])

maximum = max([entries.max() for entries in target.values()])
minimum = min([entries.min() for entries in target.values()])

print(f'Maximum: {maximum}')
print(f'Minimum: {minimum}')

for dataType, entries in target.items():
    print(f'Saving {dataType} spectrograms {entries.shape}...')
    numpy.save(
        os.path.join(config.sourceDir, f'input.{dataType}.npy'),
        utils.normalize(entries, maximum, minimum),
    )

numpy.save(
    os.path.join(config.sourceDir, f'input.bound.npy'),
    numpy.stack([maximum, minimum]),
)
