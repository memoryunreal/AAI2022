import time

import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from cross_entropy_dataset import FBanksCrossEntropyDataset
from cross_entropy_model import FBankCrossEntropyNet

import os
import sys
dirtrain = os.path.join(os.path.dirname(__file__), '..')
fbnet = os.path.join(dirtrain, "../")
prj = os.path.join(fbnet, "../")
sys.path.insert(0, dirtrain)
from pt_util import restore_objects, save_model, save_objects, restore_model
fbnet = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, fbnet)
from demo.preprocessing import get_fbanks,extract_fbanks
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
import librosa
import python_speech_features as psf
import os
import sys

fbnet = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, fbnet)
from demo.preprocessing import get_fbanks,extract_fbanks

# class FBanksTestDataset(Dataset):
#     def __init__(self, root):
#         # self.dataset_folder = DatasetFolder(root=root, loader=FBanksCrossEntropyDataset._npy_loader, extensions='.npy')
#         self.dataset_folder = DatasetFolder(root=root, loader=FBanksCrossEntropyDataset._npy_loader, extensions=('.flac'))
#         self.len_ = len(self.dataset_folder.samples)


#     @staticmethod
#     def _npy_loader(path):

#         sample = extract_fbanks(path)
#         sample = np.moveaxis(sample, 0, 2)

#         #sample = np.load(path)
#         assert sample.shape[0] == 64
#         assert sample.shape[1] == 64
#         assert sample.shape[2] == 1

#         sample = np.moveaxis(sample, 2, 0)  # pytorch expects input in the format in_channels x width x height
#         sample = torch.from_numpy(sample).float()

#         return sample

#     def __getitem__(self, index):
#         return self.dataset_folder[index]

#     def __len__(self):
#         return self.len_



def test(model, device, test_loader, log_interval=None):
    test_dir = test_loader[0]
    test_seqs = test_loader[1]

    pred = []
    with torch.no_grad():
        # for batch_idx, (x, y) in enumerate(tqdm.tqdm(test_loader)):
        for id, sp in enumerate(tqdm.tqdm(test_seqs)):
            x = extract_fbanks(os.path.join(test_dir, sp))
            x = np.moveaxis(x, 0, 2)


            x = np.moveaxis(x, 2, 0)  # pytorch expects input in the format in_channels x width x height
            x = x.reshape(1,1,64,64)
            x = torch.from_numpy(x).float()
            x = x.to(device)
            out = model(x)
            out = torch.argmax(out, dim=1)
            # test_loss_on = model.loss(out, y).item()
            # losses.append(test_loss_on)
            pred.append([sp, out])
            # accuracy += torch.sum((pred == y))s

    return pred


def main():
    model_path = os.path.join(prj, 'checkpoints/clean_100/95.pth')
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', device)

       
    dataset = os.path.join(prj, '../dataset/LibriSpeech-SI')

    # kwargs = {'num_workers': 0,
            #  'pin_memory': True} if use_cuda else {}        
    fbanks_test = os.path.join(dataset, 'test')
    # test_dataset = FBanksTestDataset(fbanks_test)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, **kwargs)
    test_dataset = os.listdir(fbanks_test)
    test_dataset.sort()
    # test_dataset = test_dataset[:10]
    test_loader = [fbanks_test, test_dataset]
    model = FBankCrossEntropyNet(reduction='mean').to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
  
    pred = test(model, device, test_loader)
    resultfile = os.path.join(prj, 'result_95.txt')
    with open(resultfile, 'w') as f:
        for i in range(len(pred)):
            line = '{} spk{:03d}'.format(pred[i][0], int(pred[i][1])+1)
            f.writelines(line)
            f.writelines("\n")

if __name__ == '__main__':
    main()
