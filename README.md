# CNN-RNN-for-Multiclass-Classification-on-SST-dataset

This is a pytorch implementation of multiclass sentiment analysis on SST dataset. In here, both CNN and RNN model implementation is shown
here.

# File Description
| File Name               | Description                                            | 
| ------------------------|:------------------------------------------------------:| 
| model.py                | Classification Models                                  | 
| utils.py                | Loading dataset, model fit, model test and visualize   |  
| train.py                | Training and testing models                            |   
| sample_execution.ipynb  | sample execution                                       |
| run.sh                  | Bash file for running this program in 6 configuration  |

# Usage
This repository can be donwloaded as 

```
git clone https://github.com/aminul-huq/CNN-RNN-for-Multiclass-Classification-on-SST-dataset.git
bash run.sh
```
Tested under python 3.6 and pytorch 1.0

# Arguments


| Command        | Default_value        | 
| -------------  |:--------------------:| 
| --model        | 0                    | 
| --lr           | 0.001                |  
| --itr          | 5                    |   
| --dropout      | 0.5                  |
| --hidden_dim   | 256                  |  
| --device       | "cpu"                | 
| --n_filters    | 100                  |
| --filter_size  | 3                    |  
