# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base /home/inho/latent-diffusion/configs/latent-diffusion/txt2img-finetune.yaml -t --gpus 0,1,2,3, --name "simple_prompt_image_space" #scale:10 bs:20 iter:14000

# CUDA_VISIBLE_DEVICES=4,5 python main.py --base /home/inho/latent-diffusion/configs/latent-diffusion/txt2img-finetune.yaml -t --gpus 0,1, --name "simple_prompt_train_latent_space" #scale:10 bs:20 iter:14000

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base /home/inho/latent-diffusion/configs/latent-diffusion/txt2img-finetune.yaml -t --gpus 0,1,2,3, --name "simple_prompt_test_latent_space_random" #scale:10 bs:20 iter:14000
