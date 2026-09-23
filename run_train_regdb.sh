CUDA_VISIBLE_DEVICES=6,2,3,4 \
python train_regdb.py --dataset regdb_rgb --batch-size 64 --arch agw --epochs 100 --lr 0.00035 \
--iters 100 --num-instances 16 \
--data-dir "/home/liyongxiang/code/ReID/dataset/RegDB" \
--logs-dir "/home/liyongxiang/code/ReID/USL-VI/training_logs/RegDB" \
--trial 1

# trial: 1,2,3,4,5,6,7,8,9,10
