import lpips
import os
import numpy as np
from PIL import Image
import torchvision

def score(path1, path2):
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    img_list = os.listdir(path1)
    loss_list = []
    totensor = torchvision.transforms.ToTensor()
    for i, img_name in enumerate(img_list):
        if i%1000 == 0:
            print(i)
        img_path1 = os.path.join(path1,img_name)
        img_path2 = os.path.join(path2,img_name)
        
        img1 = 2*totensor(Image.open(img_path1)).cuda() - 1.0
        img2 = 2*totensor(Image.open(img_path2)).cuda() - 1.0
        loss = loss_fn_alex.forward(img1, img2)
        loss_list.append(loss.detach().item())
    print("total "+str(len(loss_list))+' images pair')
    loss_list = np.array(loss_list)
    print("lpips : " + str(loss_list.mean()))

# score('/home/inho/latent-diffusion/results/baseline_prompt_scale5/samples','/home/inho/latent-diffusion/results/baseline_scale5/samples')
score('/home/inho/latent-diffusion/results/img_space_scale5/samples','/home/inho/latent-diffusion/results/img_space_prompt_scale5/samples')
score('/home/inho/latent-diffusion/results/latent_space_test_scale5/samples','/home/inho/latent-diffusion/results/latent_space_test_prompt_scale5/samples')