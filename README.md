# Image_Inpainting
Imag inpainting using Partial convolution 

#to train the model use the following command. Refer to args list in train.py
CUDA_VISIBLE_DEVICES=0 python3 train.py

#To test the model results
CUDA_VISIBLE_DEVICES=0 python3 test.py --snapshot "./snapshots/default/ckpt/1050000.pth" --location of trained model file
