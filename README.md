# Raw Audio End-to-End Deep Learning Architectures for Sound Event Detection
This is the accompanying Github repo for the thesis "Raw Audio End-to-End Deep Learning Architectures for Sound Event Detection" by Arvid Falch, UiO 2023. 

In this repo all the models trained for the thesis can be downloaded, as well as instructions for training and evaluating the models. 

## Models

All the models trained and described in the thesis can be downloaded from this [Dropbox link](https://www.dropbox.com/scl/fo/omimk5k680nqo62b5l24p/h?dl=0&rlkey=t3amz8fmstdxtboji83l2je90). 

## Training dataset
The training dataset can be downloaded from this [Dropbox link](https://www.dropbox.com/scl/fo/px2pohl0vzek7qn66hqwc/h?dl=0&rlkey=4lm91avcc4n4lqvw01jwczuui). Download all the files in the folder. The dataset has been split into four training sets and four validation sets, which are used in the training loop to avoid filling up the GPU memory. The datasets are saved as .npz files. Keep them in the same folder as the Train.py file. 

Training dataset can also be downloaded and constructed from the [DESED 2019 task 4 repo](https://github.com/turpaultn/DESED), follow the instructions on that page. However to use the training script in this repo, download the dataset from the Dropbox link.  

## Evaluation dataset
The evaluation dataset can be downloaded from this [Dropbox link](https://www.dropbox.com/scl/fo/yms4wsqj97mgwnv2wkdr3/h?dl=0&rlkey=698xzssz2rcuqe8w0wozc05kf). Keep in the same folder as the evaluation scripts.  

## Training

To train models, use Train.py.   
The following arguments are used:  
`--batchsize 10` sets the batchsize. 10 is default.   
`--epochs 100` sets the number of epochs to train. Note that each epoch is split into 4 mini-epochs while training. This argument sets the number for the total epochs.   
`--model_name testmodel` provide a name for your model, if you want to load an existing model you must provide the correct name.   
`--load_model 0` 0 for False, 1 for True.  
`--frame_size 256` gives the analysis frame size in samples, 64 or 256.   
`--model_type 0` decides the type of model architecture to train. 0 = RAW2D, 1 = LOG2D, 2 = RAW1D, 3 = LOG1D, and 4 = PURE1D (which only works on analysis frame size of 256). 

### Example
To train a RAW2D from scratch on analysis frame size of 256 for 100 epochs: 
`python Train.py --batchsize 10 --epochs 100 --model_name testRAW2D --frame_size 256  --model_type 0`

## Evaluation
For all evaluation scripts, install the following two packages:  
`pip install sed_eval` and `pip install psds-eval`.  
The Eval_256.ipynb is the main evaluation notebook, and can be used for all new models and all pretrained models working with analysis frame size of 256. The original pretrained models for analysis frame size of 64 was constructed using TimeDistributed wrappers, and therefore demands another evaluation script, so use the notebook Eval_64.ipynb in those instances.  
Make sure to set the path to the evaluation set and pretrained models in the notebooks. 
