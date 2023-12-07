from cleanfid import fid

gt_path = "/hub_data2/inho/data/coco/val2017"
sample_root = "/home/inho/latent-diffusion/results/"
model = ['img_space_scale5', 'img_space_prompt_scale5'] #'baseline_scale5', 'baseline_prompt_scale5', 'latent_space_test_random', 'latent_space_test_random_prompt', 'latent_space_test_random_scale5', 'latent_space_test_random_prompt_scale5'] #'latent_space_test_scale5','latent_space_test_prompt_scale5']#,'latent_space_test_scale15','latent_space_test_prompt_scale15']

for m in model:
    print("==================")
    print(m)
    sample_path = sample_root + m + '/samples'
    score = fid.compute_fid(sample_path, gt_path)
    print("FID : "+str(score))
    print("==================\n")



sample_path1 = sample_root + 'img_space_scale5/samples'
sample_path2 = sample_root + 'img_space_prompt_scale5/samples'
score = fid.compute_fid(sample_path1, sample_path2)
print(score)
