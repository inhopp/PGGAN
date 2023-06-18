# PGGAN
PGGAN from scratch (pytorch)

> [Paper Review](https://inhopp.github.io/paper/Paper20/)

| Sample 1 | Sample 2 |
|:-:| :-: |
| ![img_8](https://github.com/inhopp/inhopp/assets/96368476/25bf03f1-6a24-48fc-825f-636e0e4c3b54) | ![img_3](https://github.com/inhopp/inhopp/assets/96368476/b60ec28f-3025-4756-9b82-4150136abdbf)
 |


## Repository Directory 

``` python 
├── PGGAN
        ├── datasets
        │    
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── option.py
        ├── model.py
        ├── train.py
        ├── inference.py
        └── README.md
```

- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get item
- `model.py` : Define block and construct Model
- `option.py` : Environment setting

<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/PGGAN.git
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --lr {}(default: 0.0002) \
    --n_epoch {}(default: 30) \
    --num_workers {}(default: 4) \
```

### testset inference
``` python
python3 inference.py
    --device {}(defautl: cpu) \
    --num_workers {}(default: 4) \
```


<br>


#### Main Reference
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master