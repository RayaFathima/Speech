import config
import numpy
import os
import torch

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataType):
        self.data = numpy.load(
            os.path.join(config.sourceDir, f'input.{dataType}.npy'),
            mmap_mode = 'r',
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        source = numpy.expand_dims(
            sample[0].copy(),
            axis = 0,
        )
        target = sample.copy()[1:]
        return torch.from_numpy(source), torch.from_numpy(target)

class AudioMood(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 256, kernel_size = 5, stride = (2, 3)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 5, stride = 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 5, stride = 2),
            torch.nn.LeakyReLU(),
        )
        self.stage2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 19136, out_features = 2048),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features = 2048, out_features = 2048),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features = 2048, out_features = 19136),
            torch.nn.LeakyReLU(),
        )
        self.stage3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels = 128, out_channels = 512, kernel_size = 5, stride = 2, output_padding = (0, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 5, stride = 2, output_padding = (1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 5, stride = (2, 3), output_padding = (1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels = 128, out_channels = 4, kernel_size = 5, padding = 2),
            torch.nn.Sigmoid(),
        )
    def forward(self, data):
        output1 = self.stage1(data)
        # print(output1.shape)
        output2 = self.stage2(output1)
        # print(output2.shape)
        output3 = self.stage3(
            torch.cat([output1, output2.view(output1.shape)], dim = 1),
        )
        # print(output3.shape)
        return output3
