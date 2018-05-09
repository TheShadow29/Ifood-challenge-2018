from __future__ import division
import os
import numpy 
import csv
import shutil
import pdb

if not os.path.exists('../../Training_Data/'):
    os.makedirs('../../Training_Data/')

with open('../../annot/train_info.csv','r') as f:
    train_files = list(csv.reader(f,delimiter=','))

for i in range(211):
    os.makedirs('../../Training_Data/'+str(i))

for i in range(len(train_files)):
    #print('../../train_set/'+train_files[i][0])
    #print(str(class_list[int(train_files[i][1])])) 
    #pdb.set_trace()
    print('../../train_set/'+str(train_files[i][0]))
    print('../../Training_Data/'+str(train_files[i][1]))
    shutil.copy('../../train_set/'+str(train_files[i][0]),'../../Training_Data/'+str(train_files[i][1]))

    print(i)
