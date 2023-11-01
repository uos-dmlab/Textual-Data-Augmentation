python3 ./modules/main.py \
--dataset_name $dataset \
--augment_ratio $augment_ratio \
--data_ratio $data_ratio \
--augment_method None \
--balanced 1 \
--soft_label 0.2 \
--project_name small_balanced_$dataset