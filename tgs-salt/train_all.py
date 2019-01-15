from train_s1 import train_s1
from train_s2 import train_s2
from train_s3 import train_s3
from train_vat import train_vat

for i in range(1,4):
    print(f'********* FOLD {i} STAGE 1 *********')
    train_vat(i, disable_progress=True)

# for i in range(1,4):
#     print(f'********* FOLD {i} STAGE 2 *********')
#     train_s2(i, disable_progress=True)
