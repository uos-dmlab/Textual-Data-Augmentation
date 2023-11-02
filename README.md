# **Textual-Data-Embedding-Minsuk-**

This repository contains the base code for CuboidGAN, an efficient TextCuboid data augmentation method.  
Follow the Environment Setting and Train/Inference code to perform augmentation and train the classifier.  

## Environment Setting

```bash
pip3 install -r requirements.txt
```

## Train/Inference CuboidGAN

```bash
python3 ./modules/main.py \
    --dataset_name reuters \
    --augment_ratio 0.5 \
    --data_ratio 1 \
    --augment_method gan \
    --balanced 1 \
    --soft_label 0.2
```

## Reference Codes
- [Easy Data Augmentation Repository](https://github.com/jasonwei20/eda_nlp) with minor changes.
- [Marian MT Huggingface](https://huggingface.co/docs/transformers/model_doc/marian) for Back Translation.