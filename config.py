import os

audioRate = 24000
frameSize = 2048
hopLength = 512

dataMoods = ['Neutral', 'Angry', 'Happy', 'Sad', 'Surprise']
dataTypes = ['train', 'test', 'evaluation']
sourceDir = 'source'
targetDir = 'target'

modelPath = os.path.join(targetDir, 'model.pt')
batchSize = 20
trainings = 25
