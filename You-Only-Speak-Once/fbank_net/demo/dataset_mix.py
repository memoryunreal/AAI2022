import os
import shutil

dataset_clean = "/home/AAI/dataset/LibriSpeech-SI/train/"

dataset_noise = "/home/AAI/dataset/LibriSpeech-SI/noise/"

noise_sample =os.listdir(dataset_clean)
noise_sample = os.listdir(dataset_noise)

n_noise_sample = len(noise_sample)

seq_list = [os.path.join(dataset_clean, i) for i in os.listdir(dataset_clean)]
n_clean_class = len(seq_list)

noise_averge = int(n_noise_sample / n_clean_class)


for i in range(n_clean_class):
    for n in range(noise_averge):
        shutil.copy2(os.path.join(dataset_noise, noise_sample[n+i]), os.path.join(seq_list[i], noise_sample[n+i]))

