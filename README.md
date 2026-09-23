## Dataset Preprocessing
Convert the dataset format (like Market1501).
```shell
python prepare_sysu.py   # for SYSU-MM01
python prepare_regdb.py  # for RegDB
```
You need to change the file path in the `prepare_regdb.py`.

## Training
```shell
./train_regdb.sh  # for RegDB
```
Two training stages are included and you need to specify the training stage by commenting another stage's `main_worker` like this:
```python
main_worker_stage1(args,log_s1_name) # Stage 1
main_worker_stage2(args,log_s1_name,log_s2_name) # Stage 2
```

## Test
```shell
./test_regdb.sh   # for RegDB
```
