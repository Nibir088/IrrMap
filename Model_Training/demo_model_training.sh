python train.py --data_path ../train_test_split/LandSat/IrrMap_combined.yaml \
                --source landsat \
                --input_types "image,ndvi" \
                --epochs 5 \
                --batch_size 32 \
                --num_classes 4 \
                --lr 0.0005 \
                --devices -1 \
                --strategy ddp