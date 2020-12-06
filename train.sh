export CUDA_VISIBLE_DEVICES=2
nohup python -u train.py --batch_size 10 \
		--model vgg \
        --lr 5e-4 >train.log 2>&1 &
        #--resume "./weights/dsfd_vgg_0.880.pth"