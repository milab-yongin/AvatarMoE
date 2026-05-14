# datasets=("zjumocap_377_mono" "zjumocap_386_mono" "zjumocap_387_mono" "zjumocap_392_mono" "zjumocap_393_mono" "zjumocap_394_mono")
# datasets=("zjumocap_377_refine" "zjumocap_386_refine" "zjumocap_387_refine" "zjumocap_392_refine" "zjumocap_393_refine" "zjumocap_394_refine")
# datasets=("ps_female_3" "ps_female_4" "ps_male_3" "ps_male_4")
datasets=("zjumocap_377_mono")

sequences=(0 1 2 3 4 5 6 7)
for dataset in "${datasets[@]}"; do
    for seq in "${sequences[@]}"; do
        echo "Running: dataset=$dataset, seq=$seq"
        # zju_mono, zju_refine
        python render.py mode=predict dataset=$dataset dataset.predict_seq=$seq

        # pps
        # python render.py mode=predict option=iter15k pose_correction=none dataset=$dataset dataset.predict_seq=$seq
    done
done
