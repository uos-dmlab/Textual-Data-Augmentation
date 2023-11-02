# **Textual-Data-Embedding-Minsuk-**

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