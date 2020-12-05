export CUDA_VISIBLE_DEVICES=1
python train.py --batch_size 4 \
		--model vgg \
        --lr 5e-4 \
        --resume "./weights/dsfd_vgg_0.880.pth"