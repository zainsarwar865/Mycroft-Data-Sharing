#!/bin/bash

dat_path=/IOT-data/IoT-CVS/
code_dir=../scripts
results_path=./results/
# echo $dat_path
########    setting up  ###############
SET_UP_DATA=false # split into benign and attack data and create train, validation and test set
CREATE_FAKE_DO=false    # creating fake DO
CREATE_FAKE_MT=false # creating fake MT
CALCULATE_BINNING=false
RUN_DECISIONTREE_DHARD_DO=false # using decisiontree to see which  DO data points are the same path as Dhard
DIVERSE_SHARE=false
EXPERIMENT_SHARE=true


# whether we only check 1 DO or we search through all DO
search_mode=pairwise #all   # pairwise is for sharing between 1 MT and 1 DO, #all is for 1 MT and search data across all DO
# method of binning method to search for similar data to Dhard, whether it is based on statistical distribution of both Dhard and DO or only DO
binning_mode=dharddo # do , dharddo use information from both dhard and do to generate bins, do only use information from do to generate bins
MT="1Part"
DO=CnC_3
sharing_mode=full #others include  nosharing, enola, equ_enola, enola-sample_10, enola-sample_20, enola-sample-50, enola-sample_100, enola-sample_200, enola-sample_500, random-sample_10, random-sample_20, random-sample_100, random-sample_200, random-sample_500
diverse_mode=binning #decisiontree # none
diverse_num=5 # number of samples from the diverse dataset (not selected by enola to be shared in addition to data shared by enola)
model_name=DecisionTree


# # Setup the data by sorting them into attacks and create Fake DOs where we can get data from
# #############################################################################################
if [ "$SET_UP_DATA" = true ]

then
    train_test_split=0.8
    seed=0
    echo $SET_UP_DATA
    cd ${code_dir}/sort_split_data
    echo "$PWD"
    python3 sort_data_into_attacks.py \
    --data_dir=$dat_path \
    
    python3 train_val_test_split.py \
    --data_dir=$dat_path \
    --split_ratio=$train_test_split \
    --seed=$seed \

fi


# # Setup Fake DO when the data is too homogeneous
# #############################################################################################
if [ "$CREATE_FAKE_DO" = true ]

then
    cd ${code_dir}/create_fake_DO
    echo "$PWD"
    python3 generate_fake_DO.py \
    --data_dir=$dat_path \
    
fi


# # Setup Fake MT
# #############################################################################################
if [ "$CREATE_FAKE_MT" = true ]

then
    cd ${code_dir}/create_fake_MT
    echo "$PWD"
    python3 generate_fake_MT.py \
    --data_dir=$dat_path \
    
fi


# # calculate the binning distance 
# #############################################################################################  
if [ "$CALCULATE_BINNING" = true ]

then
    cd ${code_dir}/enola
    echo "$PWD"
    python3 calculate_binning_distance.py \
    --save_path=${dat_path}Binning \
    --MT_path=${dat_path}Dhard \
    --DO_path=${dat_path}FakeDO \
    --MT=$MT \
    --DO=$DO \
    --search_mode=$search_mode \
    --binning_mode=$binning_mode \

    
fi





# # run decisiontree on DO and Dhard
# #############################################################################################  
if [ "$RUN_DECISIONTREE_DHARD_DO" = true ]

then
    cd ${code_dir}/enola
    echo "$PWD"
    python3 run_decisiontree_DO_dhard.py \
    --save_path=${dat_path}DecisionTree \
    --Dhard_path=${dat_path}Dhard \
    --DO_path=${dat_path}FakeDO \
    --MT=$MT \
    --DO=$DO \
    --search_mode=$search_mode \

    
fi





# # # creating the divserse samples besides enola data
# # #############################################################################################  
if [ "$DIVERSE_SHARE" = true ]

then
    cd ${code_dir}/enola
    echo "$PWD"
    python3 diversify_share_data.py \
    --save_path=${results_path} \
    --binning_path=${dat_path}Binning \
    --diverse_path=${dat_path}Diverse \
    --MT=$MT \
    --DO=$DO \
    --search_mode=$search_mode \
    --binning_mode=$binning_mode \
    --diverse_mode=$diverse_mode \
    --diverse_num=$diverse_num \


    
fi







# # Sharing data
# #############################################################################################

if [ "$EXPERIMENT_SHARE" = true ]

then
    cd ${code_dir}/share_data
    echo "$PWD"
    python3 evaluate_share_performance.py \
    --save_path=${results_path} \
    --dat_path=${dat_path} \
    --binning_path=${dat_path}Binning \
    --diverse_path=${dat_path}Diverse \
    --MT=$MT \
    --DO=$DO \
    --search_mode=$search_mode \
    --sharing_mode=$sharing_mode \
    --binning_mode=$binning_mode \
    --diverse_mode=$diverse_mode \
    --diverse_num=$diverse_num \
    --model_name=$model_name \
    
fi



# # THE END