### Code for the paper Mycroft: Towards Effective and Efficient External Data Augmentation

#### Mycroft is a data sharing protocol which uses gradient and feature similarity based techniques to establish the utility of a dataset for a particular model in terms of external data augmentation


### Dataset processing scripts
The code for processing the datasets we use in the paper can be found under : 
`src/Enola/scripts/utils/processing_datasets/`

The final output will be a pandas dataframe which can be directly provided to a pytorch dataloader


### Constructing DHard
DHard is the portion of the test set which a model trainer performs poorly on and would like to improve performance on. This is the subset which gets shared with the DO. 

The script `src/Enola/scripts/utils/create_perfect_dhard.ipynb` can be used to create this subset

### Model training

Our framework allows training the model trainer (MT) original model, the data owner's (DO) model and the final version of the MT's model trained on the newly acquired data from the DO.

### Mycroft Pipeline

The entire pipeline for Mycroft follows the following format which is also provided in our bash scripts in `src/Enola/bash_scripts/v1`

1 - Setting up the directories where data / configs / model checkpoints and outputs will be stored
2 - Constructing data splits
3 - Training the baseline model for the MT
4 - Constructing Dhard for the MT's model from the val set
5 - Train the DO's model
6 - Run Mycroft using Gradient Matching / Feature Matching / FuncFeat
7 - Constrcut D_useful (the shared dataset)
8 - Retrain the MT's model with it - (training from scratch or finetuning)
9 - Evaluating the new model



