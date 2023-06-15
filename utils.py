import config
import librosa
import math
import matplotlib.pyplot as plt
import noisereduce
import numpy

def load(path):
    signal, sampleRate = librosa.load(path, sr = config.audioRate)
    return noisereduce.reduce_noise(
        y = signal,
        sr = sampleRate,
        n_fft = config.frameSize,
        hop_length = config.hopLength,
    )

# uniformization
def signalSizeBounds(data):
    print('Computing size limits...')
    max_len = -math.inf
    min_len = math.inf
    for entries in data.values():
        for group in entries:
            for signal in group:
                max_len = len(signal) if len(signal) > max_len else max_len
                min_len = len(signal) if len(signal) < min_len else min_len
    return max_len, min_len

def padSignal(signal, length):
    return numpy.append(
        signal,
        numpy.zeros(
            length - len(signal),
            dtype = numpy.float32,
        ),
    )

# conversion
def spectrogram(signal):
    mel = librosa.feature.melspectrogram(
        y = signal,
        sr = config.audioRate,
        n_fft = config.frameSize,
        hop_length = config.hopLength,
    )
    # return librosa.power_to_db(mel, ref = numpy.min)
    return librosa.power_to_db(mel)

def invertogram(spectrogram):
    mel = librosa.db_to_power(spectrogram)
    return librosa.feature.inverse.mel_to_audio(
        M = mel,
        sr = config.audioRate,
        n_fft = config.frameSize,
        hop_length = config.hopLength,
    )

# scaling
def normalize(data, maximum, minimum):
    return (data - minimum) / (maximum - minimum)

def rebuilder(maximum, minimum):
    def build(data):
        return [
            value * (maximum - minimum) + minimum
            for value in data
        ]
    return build

# peeking
def evaluateSample(model, loader):
    model.eval()
    source, target = next(iter(loader))
    output = model(source[0:1])
    return [
        value.detach().numpy()
        for value in [target[0], output[0]]
    ]

def contrastSample(values, savePath = None):
    length = len(values)
    points = max([len(value) for value in values])
    for y in range(points):
        for x in range(length):
            plt.subplot(points, length, length * y + x + 1)
            librosa.display.specshow(values[x][y])
    if (savePath != None):
        plt.savefig(savePath, bbox_inches = 'tight', dpi = 200)
    plt.show()

