# datasets=("zjumocap_377_mono" "zjumocap_386_mono" "zjumocap_387_mono" "zjumocap_392_mono" "zjumocap_393_mono" "zjumocap_394_mono")
# datasets=("zjumocap_377_refine" "zjumocap_386_refine" "zjumocap_387_refine" "zjumocap_392_refine" "zjumocap_393_refine" "zjumocap_394_refine")
# datasets=("ps_female_3" "ps_female_4" "ps_male_3" "ps_male_4")

for dataset in "${datasets[@]}"; do
    echo "Running: dataset=$dataset"
    
    # zju_mono, zju_refine
    python render.py mode=test dataset.test_mode=view dataset=$dataset

    # pps
    # python render.py mode=test dataset.test_mode=pose option=iter15k pose_correction=none dataset=$dataset
done