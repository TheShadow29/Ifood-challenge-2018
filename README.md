# Ifood-challenge-18
Trying the INaturalist Challenge 2018
### Links:
+ Competition Link: https://www.kaggle.com/c/ifood2018/

# Code Files:
+ `code/Preprocessing`: Includes helper functions to analyse the data. However we later found that the same could be achieved in much fewer lines of codes using `pandas` library.
+ `data`: Folder to contain all training files etc.
+ `pyt_code`: Pytorch Code all using FastAI library which can be found here : https://github.com/fastai/fastai. Since we also ran the same code for the iFungi dataset we had some duplicates. We plan to clean these up in some time. We mention the most important files (pardon the bad naming sense):
  + `Acc_95.py`: Can be run as is if the csv files are appropriately structured. Can use any of the existing architectures of Fastai as is. This includes (but not limited to) resnext50, resnext101, resnet50, inception_4. 
  + `Acc_senet_95.py`, `Acc_pnasnet_95.py`: A few modifications are required to get models provided in the fantastic repository provided in https://github.com/Cadene/pretrained-models.pytorch. Similar thing works for dual path networks.
  + `Ensemble.ipynb`: Mainly used for calculating appropriate weights to maximize scores on validation set. 
The other files were some other random experiments which we tried out but didn't give promising results.

Our final solution was an ensemble of resnext101, inceptionv4, inception-resnetv2, densenet161, se_resnext_101, nasnet. Some models were trained multiple times as well. 
