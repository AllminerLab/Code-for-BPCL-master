
## Some tips:

1. Input format:

 - For each basket sequence, baskets {b_i} are separated by '|', e.g., b_1|b_2|b_3|...|b_n
 - For each basket b_i, items {v_j} are separated by a space ' ', e.g., v_1 v_2 v_3 ... v_m

2. How to train:
 - Step 1: Run cmatrix_generator.py to generate correlation matrix. 
 - Step 2: Run main.py to train the BPCL model.


## Requirements

- Python == 3.6
- Tensorflow == 1.14
- scipy.sparse == 1.3.0


## Statement 
The code is modified based on the code of Beacon.

Reference:

Ting-Ting Su, Zhen-Yu He, Man-Sheng Chen and Chang-Dong Wang. "Basket Booster for Prototype-based Contrastive Learning in Next Basket Recommendation", ECML PKDD2022
