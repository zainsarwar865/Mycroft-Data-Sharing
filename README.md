### Code for the paper Mycroft: Towards Effective and Efficient External Data Augmentation

#### Mycroft is a data sharing protocol which uses gradient and feature similarity based techniques to establish the utility of a dataset for a particular model in terms of external data augmentation

## For vision datasets
### Dataset processing scripts
The code for processing the datasets we use in the paper can be found under : 
`src/Mycroft/scripts/utils/processing_datasets/`

The final output will be a pandas dataframe which can be directly provided to a pytorch dataloader


### Constructing DHard
DHard is the portion of the test set which a model trainer performs poorly on and would like to improve performance on. This is the subset which gets shared with the DO. 

The script `src/Mycroft/scripts/utils/create_perfect_dhard.ipynb` can be used to create this subset

### Model training

Our framework allows training the model trainer (MT) original model, the data owner's (DO) model and the final version of the MT's model trained on the newly acquired data from the DO.

### Mycroft Pipeline

The entire pipeline for Mycroft follows the following format which is also provided in our bash scripts in `src/Mycroft/bash_scripts/v1`

1 - Setting up the directories where data / configs / model checkpoints and outputs will be stored
2 - Constructing data splits
3 - Training the baseline model for the MT
4 - Constructing Dhard for the MT's model from the val set
5 - Train the DO's model
6 - Run Mycroft using Gradient Matching / Feature Matching / FuncFeat
7 - Constrcut D_useful (the shared dataset)
8 - Retrain the MT's model with it - (training from scratch or finetuning)
9 - Evaluating the new model


## For Tabular dataset

### Processing raw network data
If the network data from MT and DO is in raw format, it must be processed into a tabular format. To achieve this, you can use the helper functions available in the src/Mycroft-IoT/scripts/Supplement folder:

`nfstream_helper.py` - Converts pcap files into tabular flow statistics.
`read_log.py` - Reads and processes label files.
`merge_data.py` - Labels the flows using the processed labels and flow statistics, based on source/destination IP addresses, flow timing, etc., resulting in a CSV file with tabular features and labels.

### Creating Synthetic DO and MT
The dataset includes multiple data captures, each containing both benign and attack traffic. To simulate realistic scenarios, the data is split and recombined to create synthetic DOs and MTs:

1 - Split each capture into benign and individual attack types.
2 - Generate various DOs by mixing data from different captures and attack types. Create MTs, each representing a single attack type.

### Running mycroft
Due to differing data distributions between DO and MT, adjust distance metrics to reflect the feature distributions for both:
1 - Calculate the binning distance between Dhard and the DO samples.
2 - Select samples that are likely useful for MT. DO can choose to diversify these samples rather than merely sharing those that are the closest distance to Dhard

### Evaluate the samples that is shared 
MT evaluates the samples shared by DO to determine which DO provides the most useful data.

For more details about how to control each process, please refer to `/src/Mycroft-IoT/bash_scripts/mycroft_e2e.sh`
