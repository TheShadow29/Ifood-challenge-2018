# Ifood-challenge-18
Trying the INaturalist Challenge 2018
### Links:
+ Competition Link: https://www.kaggle.com/c/ifood2018/
+ Data Link: https://github.com/karansikka1/Foodx/blob/master/README.md#data-download-and-format

## Addendum
A lot of code is taken from lesson2 of Fast.ai course.

## Code Files:
+ `code/Preprocessing`: Includes helper functions to analyse the data. However we later found that the same could be achieved in much fewer lines of codes using `pandas` library.
+ `data`: Folder to contain all training files etc.
+ `pyt_code`: Pytorch Code all using FastAI library which can be found here : https://github.com/fastai/fastai. Since we also ran the same code for the iFungi dataset we had some duplicates. We plan to clean these up in some time. We mention the most important files (pardon the bad naming sense):
  + `Acc_95.py`: Can be run as is if the csv files are appropriately structured. Can use any of the existing architectures of Fastai as is. This includes (but not limited to) resnext50, resnext101, resnet50, inception_4. 
  + `Acc_senet_95.py`, `Acc_pnasnet_95.py`: A few modifications are required to get models provided in the fantastic repository provided in https://github.com/Cadene/pretrained-models.pytorch. Similar thing works for dual path networks.
  + `Ensemble.ipynb`: Mainly used for calculating appropriate weights to maximize scores on validation set. 
The other files were some other random experiments which we tried out but didn't give promising results.

Our final solution was an ensemble of resnext101, inceptionv4, inception-resnetv2, densenet161, se_resnext_101, nasnet. Some models were trained multiple times as well. 

## Using Trained Models:
All the trained models can be found here: https://tinyurl.com/ifood-best-models. Again the nomenclature is not very good so sorry for that. `best_rxnet_u5.h5` is the best performing single model. To use any pretrained model, simply open `ipython`, run `Acc_95.py` with the correct architecture (senet, densenet, dualpath require other files which can be found in `pyt_code`) and once done, type `learn.load(weightsfile)` without the `.h5` extension. Also `weightsfile.h5` should be put inside the directory `data/models/`. It might be helpful to checkout saving and loading weights using Fastai library (which is what we use here).

## Data structure:
Inside the `data` folder, append all the validation images to the training set and use corresponding validation indices which correspond to the validation files. Test can be kept untouched.
