text_path="/hub_data2/inho/data/coco/annotations/val_txt"
sample_root="/home/inho/latent-diffusion/results/"
for model in 'latent_space_test_random_scale5' 'latent_space_test_random_prompt_scale5' #'latent_space_test_random_prompt' 'latent_space_test_random_prompt_scale5' #'baseline_scale5' 'baseline_prompt_scale5' 'latent_space_test_random' 'latent_space_test_prompt_random' 'latent_space_test_random_scale5' 'latent_space_test_prompt_random_scale5' #'baseline' 'baseline_prompt' 'img_space' 'img_space_prompt' 'latent_space_test' 'latent_space_test_prompt' 'latent_space_test_scale5' 'latent_space_test_prompt_scale5'
do
    echo ${model}
    python -m clip_score ${sample_root}${model}"/samples" ${text_path} --device cuda:0
done