dataset_root_paths = {
    "Imagenet":  "/bigstor/zsarwar/Imagenet/DF/",
    "Tsinghua": "/bigstor/zsarwar/Tsinghua/DF/",
    "OpenImages": "/bigstor/zsarwar/OpenImages/DF/",
    "food101": "/projects/bdgs/zsarwar/data/food-101/DF", 
    "uecfood256": "/bigstor/common_data/UECFOOD256/DF/",
    "upmcfood101": "/projects/bdgs/zsarwar/data/UPMC-food-101/DF",
    "ISIAFood101": "/bigstor/common_data/ISIA-Food-500/DF",
    "DogsVsWolves": "/projects/bdgs/zsarwar/data/Imagenet_2012_subsets/DF"



}
dataset_configs = {
    "Imagenet": {'train': {"dogs": "df_imagenet_dogs_train.pkl", 
                           "bottom_50": "df_imagenet_bottom50_train.pkl",
                           "full": "df_imagenet_train.pkl"},
                 'val': {"dogs": "df_imagenet_dogs_val.pkl", 
                           "bottom_50": "df_imagenet_bottom50_val.pkl",
                           "full": "df_imagenet_val.pkl"}},

    "Tsinghua": {"train": {"dogs": "df_tsinghua_train.pkl",
                           "dogs_corrupted": "df_tsinghua_train_corrupted.pkl",
                           "dogs_corrupted_labels": "df_tsinghua_train_corrupted_labels.pkl"},
                 
                 'val': {"dogs": "df_tsinghua_val.pkl",
                         "dogs_corrupted": "df_tsinghua_val.pkl",
                         "dogs_corrupted_labels": "df_tsinghua_val.pkl"}},

    "OpenImages": {"train": {"dogs": "df_oi_dogs.pkl" }},

    "food101": {"train": {"full": "df_food101_train.pkl"}, 
                'val': {"full": "df_food101_val.pkl"}},

    "uecfood256": {"train": {"full": "df_uec256_train.pkl" },
                    "val": {"full": "df_uec256_val.pkl"}},  

    "upmcfood101": {"train": {"full": "df_upmc-food-101_train.pkl" ,
                              "corrupted": "df_upmc-food-101_train_corrupted.pkl",
                               "corrupted_labels": "df_upmc-food-101_train_corrupted_labels.pkl" },
                    "val": {"full": "df_upmc-food-101_val.pkl",
                            "corrupted": "df_upmc-food-101_val.pkl",
                            "corrupted_labels": "df_upmc-food-101_val.pkl"}},

    "ISIAFood101": {"train": {"full": "df_ISIA-Food-500_train.pkl" ,
                              "corrupted": "df_ISIA-Food-500_train_corrupted.pkl",
                               "corrupted_labels": "df_ISIA-Food-500_train_corrupted_labels.pkl" },
                    "val": {"full": "df_ISIA-Food-500_val.pkl",
                            "corrupted": "df_ISIA-Food-500_val.pkl",
                            "corrupted_labels": "df_ISIA-Food-500_val.pkl"}},

    "DogsVsWolves": {"train" : {"MT_3" : "df_train_MT_3_Imagenet_8_Non-Dog-wolf-animals.pkl",
                                "DO_1" : "df_train_DO_1_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_2" : "df_train_DO_2_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_3" : "df_train_DO_3_Imagenet_48_no-wolf-dog-animals.pkl", 
                                "DO_4" : "df_train_DO_4_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_5" : "DF/df_train_DO_5_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_6" : "df_train_DO_6_Imagenet_48_including-wolf-dog-animals_natural.pkl",
                                "DO_1_corrupted" : "df_train_DO_1_Imagenet_48_no-wolf-dog-animals_corrupted.pkl",
                                "DO_1_corrupted_labels" : "df_train_DO_1_Imagenet_48_no-wolf-dog-animals_corrupted_labels.pkl"                                
                                },
                     "val" :   {"MT_3" : "df_val_MT_3_Imagenet_8_Non-Dog-wolf-animals.pkl",
                                "DO_1" : "df_val_DO_1_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_2" : "df_val_DO_1_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_2" : "df_val_DO_1_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_3" : "df_val_DO_1_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_4" : "df_val_DO_1_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_5" : "df_val_DO_1_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_6" : "df_val_DO_1_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_1_corrupted" : "df_val_DO_1_Imagenet_48_no-wolf-dog-animals.pkl",
                                "DO_1_corrupted_labels" : "df_val_DO_1_Imagenet_48_no-wolf-dog-animals.pkl", }}
    }