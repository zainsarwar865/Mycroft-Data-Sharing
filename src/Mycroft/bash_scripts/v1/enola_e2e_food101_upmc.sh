# Five main components of the Enola Simulator
CREATE_ROOT=false
CONSTRUCT_TEST_VAL_SPLIT=false
TRAIN_MT_BASELINE=true
CONSTRUCT_DHARD=false
TRAIN_DO=false
RUN_ENOLA=false
CONSTRUCT_D_USEFUL=false
TRAIN_MT_AUGMENTED=false


#echo $y
BashName=${0##*/}
#x=FileName
# main directory here
BashPath="$PWD"/$BashName
home_dir=/u/npatil/Code_Clean/Mycroft-Data-Sharing/src

# Generic params

gpu=0
seed=60
# 40, 50, 60 etc

# Setup the directory for an experiment
#####################################################################   ########################

# MT Root parameters
enola_base_dir='/work/hdd/bdgs/npatil/test/Clean'
#enola_base_dir='/work/hdd/bdgs/npatil/test/optim'
mt_dataset="food101"
mt_config="full"
mt_classes=101
# root_config --> subset
# Hash configs
root_hash_config="MT_${mt_dataset}_${mt_config}_${mt_classes}"
if [ "$CREATE_ROOT" = true ]
then
    cd ${home_dir}/Mycroft/scripts/utils
    python3 create_directories.py \
    --enola_base_dir=$enola_base_dir \
    --root_hash_config=$root_hash_config \
    
fi

#############################################################################################



#############################################################################################
# Copy bash script to root dir
cd ${home_dir}/Mycroft/scripts/utils
    python3 copy_bash.py \
    --enola_base_dir=$enola_base_dir \
    --root_hash_config=$root_hash_config \
    --bash_script_config=$BashPath


#############################################################################################

# Construct Test/Val splits
#############################################################################################

# Test/val parameters
split_ratio=1
if [ "$CONSTRUCT_TEST_VAL_SPLIT" = true ]
then 
    cd ${home_dir}/Mycroft/scripts/utils
    python3 create_test_val_split.py \
    --enola_base_dir=$enola_base_dir \
    --root_hash_config=$root_hash_config \
    --seed=$seed \
    --original_dataset=$mt_dataset \
    --original_config=$mt_config \
    --split_ratio=$split_ratio
fi

#############################################################################################



# Train MT's baseline model
#############################################################################################

# MT train parameters
batch_size=512
lr=0.03
weight_decay=2e-05
lr_warmup_epochs=2
lr_warmup_decay=0.01
label_smoothing=0.1
mixup_alpha=0.2
cutmix_alpha=0.2
random_erasing=0.1
model_ema=False
epochs=10
num_eval_epochs=1
resume=False
finetune=False
pretrained=True
freeze_layers=False
seed=$seed
num_classes=120
new_classifier=True
external_augmentation=False
test_per_class=True
original_dataset=$mt_dataset
original_config=$mt_config
model='resnet50'
trainer_type="MT_Baseline"
mt_hash_config="${trainer_type}_${original_dataset}_${original_config}_${model}_pretrained-${pretrained}_freeze-layers-${freeze_layers}_lr-${lr}_batch_size-${batch_size}_lr-warmup-epochs-${lr_warmup_epochs}_lr-warmup-decay-${lr_warmup_decay}_label-smoothing-${label_smoothing}_mixup-alpha-${mixum_alpha}_cutmix_alpha-${cutmix_alpha}_random-erasing-${random_erasing}_model-ema-${model_ema}_weight_decay-${weight_decay}_epochs-${epochs}_seed-${seed}"
mt_hash_ft_resume_config="${mt_hash_config}"
if [ "$TRAIN_MT_BASELINE" = true ]
then
    cd ${home_dir}/Mycroft/scripts
    python3 train.py \
    --gpu=$gpu \
    --enola_base_dir=$enola_base_dir \
    --root_hash_config=$root_hash_config \
    --mt_hash_config=$mt_hash_config \
    --epochs=$epochs \
    --num_eval_epochs=$num_eval_epochs \
    --arch=$model \
    --batch_size=$batch_size \
    --lr=$lr \
    --weight_decay=$weight_decay \
    --lr_warmup_epochs=2 \
    --lr_warmup_decay=0.01 \
    --label_smoothing=0.1 \
    --mixup_alpha=0.2 \
    --cutmix_alpha=0.2 \
    --random_erasing=0.1 \
    --model_ema=False \
    --resume=$resume \
    --pretrained=$pretrained \
    --freeze_layers=$freeze_layers \
    --seed=$seed \
    --num_classes=$num_classes \
    --new_classifier=$new_classifier \
    --external_augmentation=$external_augmentation \
    --test_per_class=$test_per_class \
    --original_dataset=$original_dataset \
    --original_config=$original_config \
    --trainer_type=$trainer_type
fi

#############################################################################################


# Create empirical DHard
#############################################################################################

# DHard parameters
bottom_k=20
per_class_budget=4
if [ "$CONSTRUCT_DHARD" = true ]
then 
    cd ${home_dir}/Mycroft/scripts/utils
    python3 create_dhard.py \
    --enola_base_dir=$enola_base_dir \
    --root_hash_config=$root_hash_config \
    --original_dataset=$mt_dataset \
    --original_config=$mt_config \
    --mt_hash_config=$mt_hash_config \
    --bottom_k=$bottom_k \
    --per_class_budget=$per_class_budget \
    --trainer_type=$trainer_type
fi


#############################################################################################


# Train DO's baseline model
#############################################################################################

do_seed=60


# DO train parameters
batch_size=512
lr=0.03
weight_decay=2e-05
lr_warmup_epochs=2
lr_warmup_decay=0.01
label_smoothing=0.1
mixup_alpha=0.2
cutmix_alpha=0.2
random_erasing=0.1
model_ema=False
epochs=10
num_eval_epochs=1
resume=False
finetune=False
pretrained=True
freeze_layers=False
num_classes=101 # 120 to match hash configs
new_classifier=True
external_augmentation=False
test_per_class=True
original_dataset="upmcfood101"
original_config="full"
model="resnet50"
trainer_type="DO"
do_hash_config="${trainer_type}_${original_dataset}_${original_config}_${model}_pretrained-${pretrained}_freeze-layers-${freeze_layers}_lr-${lr}_batch_size-${batch_size}_lr-warmup-epochs-${lr_warmup_epochs}_lr-warmup-decay-${lr_warmup_decay}_label-smoothing-${label_smoothing}_mixup-alpha-${mixum_alpha}_cutmix_alpha-${cutmix_alpha}_random-erasing-${random_erasing}_model-ema-${model_ema}_weight_decay-${weight_decay}_epochs-${epochs}_seed-${do_seed}"

if [ "$TRAIN_DO" = true ]
then
    cd ${home_dir}/Mycroft/scripts
     python3 train.py \
    --gpu=$gpu \
    --enola_base_dir=$enola_base_dir \
    --root_hash_config=$root_hash_config \
    --do_hash_config=$do_hash_config \
    --epochs=$epochs \
    --num_eval_epochs=$num_eval_epochs \
    --arch=$model \
    --batch_size=$batch_size \
    --lr=$lr \
    --weight_decay=$weight_decay \
    --lr_warmup_epochs=2 \
    --lr_warmup_decay=0.01 \
    --label_smoothing=0.1 \
    --mixup_alpha=0.2 \
    --cutmix_alpha=0.2 \
    --random_erasing=0.1 \
    --model_ema=False \
    --resume=$resume \
    --pretrained=$pretrained \
    --freeze_layers=$freeze_layers \
    --seed=$do_seed \
    --num_classes=$num_classes \
    --new_classifier=$new_classifier \
    --external_augmentation=$external_augmentation \
    --test_per_class=$test_per_class \
    --original_dataset=$original_dataset \
    --original_config=$original_config \
    --trainer_type=$trainer_type
fi

#############################################################################################





# ENOLA 
#############################################################################################

# ENOLA Generic
RUN_UNICOM=false
RUN_GRADMATCH=true
RUN_RANDOM=false
RUN_FULL=false
RUN_JOINT_OPTIM=false

# Per class budget
d_hard_budget=100
num_candidates=1

# Unicom parameters
top_k=10

# GradMatch parameters
start=0
end=4
jump_ckpts=1
trn_batch_size=1
val_batch_size=1
tst_batch_size=1000
model_eval_batch_size=512
data_budget=200
select_subset_every=1
per_class='True'
numclasses=101
model='resnet50'
lam2=200

# Enola hash configs
unicom_hash_config="Unicom_DO-dataset-${original_dataset}_DO-config-${original_config}_MT-dataset-${mt_dataset}-MT-config-${mt_config}-top-k-${top_k}_seed-${seed}"
gradmatch_hash_config="GradMatch_DO-dataset-${original_dataset}_DO-config-${original_config}_MT-dataset-${mt_dataset}-MT-config-${mt_config}_model-${arch}_joint_optimization-False_numclasses-${numclasses}_select-subset-every-${select_subset_every}_num-candidates-${num_candidates}_train-batch-size-${trn_batch_size}_model-eval-batch-size-${model_eval_batch_size}_start-${start}_end-${end}_jump-ckpts-${jump_ckpts}_per-class-${per_class}_seed-${seed}"
construct_random_config="Random_DO-dataset-${original_dataset}_DO-config-${original_config}_MT-dataset-${mt_dataset}_MT-config-${mt_config}_d-hard-budget-${d_hard_budget}"
construct_full_config="Full_DO-dataset-${original_dataset}_DO-config-${original_config}_MT-dataset-${mt_dataset}_MT-config-${mt_config}"
JointOptimization_hash_config="JointOptimization_DO-dataset-${original_dataset}_DO-config-${original_config}_MT-dataset-${mt_dataset}-MT-config-${mt_config}_model-${arch}_joint_optimization-True_numclasses-${numclasses}_select-subset-every-${select_subset_every}_num-candidates-${num_candidates}_train-batch-size-${trn_batch_size}_model-eval-batch-size-${model_eval_batch_size}_start-${start}_end-${end}_jump-ckpts-${jump_ckpts}_per-class-${per_class}_lam2-${lam2}_seed-${seed}"

if [ "$RUN_ENOLA" = true ]
then
    if [ "$RUN_UNICOM" = true ]
    then
        export CUDA_VISIBLE_DEVICES=$gpu
        cd ${home_dir}/unicom/unicom
        torchrun retrieval_generic_classwise.py \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --do_hash_config=$do_hash_config \
        --unicom_hash_config=$unicom_hash_config \
        --DO_dataset=$original_dataset \
        --DO_config=$original_config \
        --MT_dataset=$mt_dataset \
        --MT_config=$mt_config \
        --top_k=$top_k \
        --gpu=$gpu
        fi

    if [ "$RUN_GRADMATCH" = true ]
    then
        cd ${home_dir}/GRDM/cords/tutorial
        python3 OMP_e2e.py \
        --home_dir=$home_dir \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --do_hash_config=$do_hash_config \
        --gradmatch_hash_config=$gradmatch_hash_config \
        --gpu_id=$gpu \
        --start=$start \
        --end=$end \
        --jump_ckpts=$jump_ckpts \
        --trn_batch_size=$trn_batch_size \
        --val_batch_size=$val_batch_size \
        --tst_batch_size=$tst_batch_size \
        --model_eval_batch_size=$model_eval_batch_size \
        --num_candidates=$num_candidates \
        --select_subset_every=$select_subset_every \
        --per_class=$per_class \
        --numclasses=$numclasses \
        --arch=$model \
        --DO_dataset=$original_dataset \
        --DO_config=$original_config \
        --MT_dataset=$mt_dataset \
        --MT_config=$mt_config \
        --seed=$seed
    fi

    if [ "$RUN_RANDOM" = true ]
    then
        # Random should randomly select n samples from each class and share it
        cd ${home_dir}/Mycroft/scripts/utils
        python3 construct_retrieved_random.py \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --do_hash_config=$do_hash_config \
        --mt_hash_config=$mt_hash_config \
        --num_candidates=$num_candidates \
        --construct_hash_config=$construct_random_config \
        --MT_dataset=$mt_dataset \
        --MT_config=$mt_config \
        --DO_dataset=$original_dataset \
        --DO_config=$original_config \
        --d_hard_budget=$d_hard_budget \
        --seed=$seed
    fi

    if [ "$RUN_FULL" = true ]
    then
        cd ${home_dir}/Mycroft/scripts/utils
        python3 construct_retrieved_full.py \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --do_hash_config=$do_hash_config \
        --mt_hash_config=$mt_hash_config \
        --construct_hash_config=$construct_full_config \
        --MT_dataset=$mt_dataset \
        --MT_config=$mt_config \
        --DO_dataset=$original_dataset \
        --DO_config=$original_config \
        --seed=$seed
    fi

    if [ "$RUN_JOINT_OPTIM" = true ]
    then
        joint_optimization=True
        export CUDA_VISIBLE_DEVICES=$gpu    
        #cd ${home_dir}/unicom/unicom
        #torchrun retrieval_generic_joint.py \
        #--enola_base_dir=$enola_base_dir \
        #--root_hash_config=$root_hash_config \
        #--mt_hash_config=$mt_hash_config \
        #--do_hash_config=$do_hash_config \
        #--unicom_hash_config=$unicom_hash_config \
        #--DO_dataset=$original_dataset \
        #--DO_config=$original_config \
        #--MT_dataset=$mt_dataset \
        #--MT_config=$mt_config \
        #--top_k=$top_k \
        #--gpu=$gpu

        cd ${home_dir}/GradMatch/cords/tutorial
        python3 OMP_joint_e2e.py \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --do_hash_config=$do_hash_config \
        --JointOptimization_hash_config=$JointOptimization_hash_config \
        --unicom_hash_config=$unicom_hash_config \
        --gpu_id=$gpu \
        --start=$start \
        --end=$end \
        --jump_ckpts=$jump_ckpts \
        --trn_batch_size=$trn_batch_size \
        --val_batch_size=$val_batch_size \
        --tst_batch_size=$tst_batch_size \
        --model_eval_batch_size=$model_eval_batch_size \
        --num_candidates=$num_candidates \
        --select_subset_every=$select_subset_every \
        --per_class=$per_class \
        --numclasses=$numclasses \
        --arch=$model \
        --joint_optimization=$joint_optimization \
        --lam2=$lam2 \
        --DO_dataset=$original_dataset \
        --DO_config=$original_config \
        --MT_dataset=$mt_dataset \
        --MT_config=$mt_config \
        --seed=$seed
    fi
fi

#############################################################################################





# Construct D_useful
#############################################################################################

CONSTRUCT_UNICOM=false
CONSTRUCT_GRADMATCH=true
CONSTRUCT_JOINT_OPTIM=false

#construct unicom parameters

voting_scheme="Top_k_veto"
retrieve_diff_classes=False
#construct GradMatch parameters
checkpoint_idx=1
# Per class budget added by me
d_hard_budget=100

# hash configs for constructing d_useful
construct_unicom_config="Unicom_num-candidates-${num_candidates}_d-hard-budget-${d_hard_budget}_voting-scheme-${voting_scheme}_extra-classes-${retrieve_diff_classes}"
construct_gradmatch_config="GradMatch_checkpoint-index-${checkpoint_idx}_d-hard-budget-${d_hard_budget}_extra-classes-possible-${extra_classes_possible}"
construct_JointOptimization_config="JointOptimization_checkpoint-index-${checkpoint_idx}_d-hard-budget-${d_hard_budget}_extra-classes-possible-${extra_classes_possible}_lam2-${lam2}"


if [ "$CONSTRUCT_D_USEFUL" = true ]
then
    if [ "$CONSTRUCT_UNICOM" = true ]
    then
        cd ${home_dir}/Mycroft/scripts/utils
        python3 construct_retrieved_unicom_classwise.py \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --do_hash_config=$do_hash_config \
        --construct_hash_config=$construct_unicom_config \
        --unicom_hash_config=$unicom_hash_config \
        --DO_dataset=$original_dataset \
        --DO_config=$original_config \
        --MT_dataset=$mt_dataset \
        --MT_config=$mt_config \
        --num_candidates=$num_candidates \
        --voting_scheme=$voting_scheme \
        --seed=$seed
    fi

    if [ "$CONSTRUCT_GRADMATCH" = true ]
    then
        cd ${home_dir}/Mycroft/scripts/utils
        python3 construct_retrieved_gradmatch.py \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --do_hash_config=$do_hash_config \
        --construct_hash_config=$construct_gradmatch_config \
        --gradmatch_hash_config=$gradmatch_hash_config \
        --DO_dataset=$original_dataset \
        --DO_config=$original_config \
        --MT_dataset=$mt_dataset \
        --MT_config=$mt_config \
        --checkpoint_idx=$checkpoint_idx \
        --seed=$seed 
    fi

    if [ "$CONSTRUCT_JOINT_OPTIM" = true ]
        then
            cd ${home_dir}/Mycroft/scripts/utils
            python3 construct_retrieved_joint_optimization.py \
            --enola_base_dir=$enola_base_dir \
            --root_hash_config=$root_hash_config \
            --mt_hash_config=$mt_hash_config \
            --do_hash_config=$do_hash_config \
            --construct_hash_config=$construct_JointOptimization_config \
            --JointOptimization_hash_config=$JointOptimization_hash_config \
            --DO_dataset=$original_dataset \
            --DO_config=$original_config \
            --MT_dataset=$mt_dataset \
            --MT_config=$mt_config \
            --d_hard_budget=$d_hard_budget \
            --checkpoint_idx=$checkpoint_idx \
            --seed=$seed 
        fi
fi


#############################################################################################



# Train MT Augmented model
#############################################################################################

TRAIN_AUGMENTED_UNICOM=false
TRAIN_AUGMENTED_GRADMATCH=true
TRAIN_AUGMENTED_RANDOM=false
TRAIN_AUGMENTED_FULL=false
TRAIN_AUGMENTED_JOINT_OPTIM=false

# MT augmented parameters
batch_size=512
lr=0.006
weight_decay=2e-05
lr_warmup_epochs=2
lr_warmup_decay=0.01
label_smoothing=0.1
mixup_alpha=0.2
cutmix_alpha=0.2
random_erasing=0.1
model_ema=False
epochs=5
num_eval_epochs=1
resume=False
finetune=True
pretrained=True
freeze_layers=False
seed=$seed
num_classes=120
new_classifier=True
external_augmentation=True
test_per_class=True
model='resnet50'
augmented_dataset=$original_dataset
augmented_config=$original_config
original_dataset=$mt_dataset
original_config=$mt_config
trainer_type="MT_Augmented"
mt_hash_config="${trainer_type}_${original_dataset}_${original_config}_${model}_pretrained-${pretrained}_freeze-layers-${freeze_layers}_lr-${lr}_batch_size-${batch_size}_lr-warmup-epochs-${lr_warmup_epochs}_lr-warmup-decay-${lr_warmup_decay}_label-smoothing-${label_smoothing}_mixup-alpha-${mixum_alpha}_cutmix_alpha-${cutmix_alpha}_random-erasing-${random_erasing}_model-ema-${model_ema}_weight_decay-${weight_decay}_epochs-${epochs}_seed-${seed}_resume-${resume}"

if [ "$TRAIN_MT_AUGMENTED" = true ]
then
    if [ "$TRAIN_AUGMENTED_UNICOM" = true ]
    then
        cd ${home_dir}/Mycroft/scripts
        python3 train.py \
        --gpu=$gpu \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --mt_hash_ft_resume_config=$mt_hash_ft_resume_config \
        --do_hash_config=$do_hash_config \
        --do_augmented_hash_config=$unicom_hash_config \
        --do_augmented_construct_hash_config=$construct_unicom_config \
        --augmentation_technique=Unicom \
        --epochs=$epochs \
        --num_eval_epochs=$num_eval_epochs \
        --arch=$model \
        --batch_size=$batch_size \
        --lr=$lr \
        --weight_decay=$weight_decay \
        --lr_warmup_epochs=2 \
        --lr_warmup_decay=0.01 \
        --label_smoothing=0.1 \
        --mixup_alpha=0.2 \
        --cutmix_alpha=0.2 \
        --random_erasing=0.1 \
        --model_ema=False \
        --resume=$resume \
        --pretrained=$pretrained \
        --freeze_layers=$freeze_layers \
        --seed=$seed \
        --num_classes=$num_classes \
        --new_classifier=$new_classifier \
        --external_augmentation=$external_augmentation \
        --test_per_class=$test_per_class \
        --original_dataset=$original_dataset \
        --original_config=$original_config \
        --augmented_dataset=$augmented_dataset \
        --augmented_config=$augmented_config \
        --trainer_type=$trainer_type \
        --finetune=$finetune
    fi

    if [ "$TRAIN_AUGMENTED_GRADMATCH" = true ]
    then
        cd ${home_dir}/Mycroft/scripts
        python3 train.py \
        --gpu=$gpu \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --do_hash_config=$do_hash_config \
        --mt_hash_config=$mt_hash_config \
        --mt_hash_ft_resume_config=$mt_hash_ft_resume_config \
        --do_augmented_hash_config=$gradmatch_hash_config \
        --do_augmented_construct_hash_config=$construct_gradmatch_config \
        --augmentation_technique=GradMatch \
        --epochs=$epochs \
        --num_eval_epochs=$num_eval_epochs \
        --arch=$model \
        --batch_size=$batch_size \
        --lr=$lr \
        --weight_decay=$weight_decay \
        --lr_warmup_epochs=2 \
        --lr_warmup_decay=0.01 \
        --label_smoothing=0.1 \
        --mixup_alpha=0.2 \
        --cutmix_alpha=0.2 \
        --random_erasing=0.1 \
        --model_ema=False \
        --resume=$resume \
        --pretrained=$pretrained \
        --freeze_layers=$freeze_layers \
        --seed=$seed \
        --num_classes=$num_classes \
        --new_classifier=$new_classifier \
        --external_augmentation=$external_augmentation \
        --test_per_class=$test_per_class \
        --original_dataset=$original_dataset \
        --original_config=$original_config \
        --augmented_dataset=$augmented_dataset \
        --augmented_config=$augmented_config \
        --trainer_type=$trainer_type \
        --finetune=$finetune
    fi

    if [ "$TRAIN_AUGMENTED_RANDOM" = true ]
    then
        cd ${home_dir}/Mycroft/scripts
        python3 train.py \
        --gpu=$gpu \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --do_hash_config=$do_hash_config \
        --mt_hash_config=$mt_hash_config \
        --mt_hash_ft_resume_config=$mt_hash_ft_resume_config \
        --do_augmented_hash_config=$construct_random_config \
        --do_augmented_construct_hash_config=$construct_random_config \
        --augmentation_technique=Random \
        --epochs=$epochs \
        --num_eval_epochs=$num_eval_epochs \
        --arch=$model \
        --batch_size=$batch_size \
        --lr=$lr \
        --weight_decay=$weight_decay \
        --lr_warmup_epochs=2 \
        --lr_warmup_decay=0.01 \
        --label_smoothing=0.1 \
        --mixup_alpha=0.2 \
        --cutmix_alpha=0.2 \
        --random_erasing=0.1 \
        --model_ema=False \
        --resume=$resume \
        --pretrained=$pretrained \
        --freeze_layers=$freeze_layers \
        --seed=$seed \
        --num_classes=$num_classes \
        --new_classifier=$new_classifier \
        --external_augmentation=$external_augmentation \
        --test_per_class=$test_per_class \
        --original_dataset=$original_dataset \
        --original_config=$original_config \
        --augmented_dataset=$augmented_dataset \
        --augmented_config=$augmented_config \
        --trainer_type=$trainer_type \
        --finetune=$finetune
    fi

    if [ "$TRAIN_AUGMENTED_FULL" = true ]
    then
        cd ${home_dir}/Mycroft/scripts
        python3 train.py \
        --gpu=$gpu \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --mt_hash_config=$mt_hash_config \
        --mt_hash_ft_resume_config=$mt_hash_ft_resume_config \
        --do_hash_config=$do_hash_config \
        --do_augmented_hash_config=$construct_full_config \
        --do_augmented_construct_hash_config=$construct_full_config \
        --augmentation_technique=Full \
        --epochs=$epochs \
        --num_eval_epochs=$num_eval_epochs \
        --arch=$model \
        --batch_size=$batch_size \
        --lr=$lr \
        --weight_decay=$weight_decay \
        --lr_warmup_epochs=2 \
        --lr_warmup_decay=0.01 \
        --label_smoothing=0.1 \
        --mixup_alpha=0.2 \
        --cutmix_alpha=0.2 \
        --random_erasing=0.1 \
        --model_ema=False \
        --resume=$resume \
        --pretrained=$pretrained \
        --freeze_layers=$freeze_layers \
        --seed=$seed \
        --num_classes=$num_classes \
        --new_classifier=$new_classifier \
        --external_augmentation=$external_augmentation \
        --test_per_class=$test_per_class \
        --original_dataset=$original_dataset \
        --original_config=$original_config \
        --augmented_dataset=$augmented_dataset \
        --augmented_config=$augmented_config \
        --trainer_type=$trainer_type \
        --finetune=$finetune
    fi

    if [ "$TRAIN_AUGMENTED_JOINT_OPTIM" = true ]
    then
        cd ${home_dir}/Mycroft/scripts
        python3 train.py \
        --gpu=$gpu \
        --enola_base_dir=$enola_base_dir \
        --root_hash_config=$root_hash_config \
        --do_hash_config=$do_hash_config \
        --mt_hash_config=$mt_hash_config \
        --mt_hash_ft_resume_config=$mt_hash_ft_resume_config \
        --do_augmented_hash_config=$JointOptimization_hash_config \
        --do_augmented_construct_hash_config=$construct_JointOptimization_config \
        --augmentation_technique=JointOptimization \
        --epochs=$epochs \
        --num_eval_epochs=$num_eval_epochs \
        --arch=$model \
        --batch_size=$batch_size \
        --lr=$lr \
        --weight_decay=$weight_decay \
        --lr_warmup_epochs=2 \
        --lr_warmup_decay=0.01 \
        --label_smoothing=0.1 \
        --mixup_alpha=0.2 \
        --cutmix_alpha=0.2 \
        --random_erasing=0.1 \
        --model_ema=False \
        --resume=$resume \
        --pretrained=$pretrained \
        --freeze_layers=$freeze_layers \
        --seed=$seed \
        --num_classes=$num_classes \
        --new_classifier=$new_classifier \
        --external_augmentation=$external_augmentation \
        --test_per_class=$test_per_class \
        --original_dataset=$original_dataset \
        --original_config=$original_config \
        --augmented_dataset=$augmented_dataset \
        --augmented_config=$augmented_config \
        --trainer_type=$trainer_type \
        --finetune=$finetune
    fi


fi
#############################################################################################


# THE END

