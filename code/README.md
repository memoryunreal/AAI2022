# Environment installation
```
bash requirement.sh
```
# dataset preprocessing
- datasetpath 
8卡32G V100: /ssd3/lz/dataset/AAI/dataset/
```
cd fbank_net/demo/
python dataset_mix.py # adding noise samples to clean data

```

# Training
```
cd fbank_net/model_training/cross_entropy_pre_training/

python clean_100.py # train with all clean data for 100 epoches, (submmited resutls)
python mix_dataset_train.py # train with noise-clean hybrid dataset for 100 epoches
python clean20_noise80.py # train with clean data for 20 epoches, then train with noise-clean hybrid dataset for 80 epoches
```

# Testing
```
cd fbank_net/model_training/cross_entropy_pre_training/
python sp_re_test.py 

# line 95  specify the checkpoints path
# line 102 specify the training data
# line 114 specify the output result file name 
```