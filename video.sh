
Datasets=("davis_tempered" "SegTrackv2_tempered"
        "youtube_tempered")

rate=5
echo Frame Rate: $rate

for dat in "${Datasets[@]}";
do

        v_dir="./data/${dat}/vid"
        gt_dir="./data/${dat}/gt_mask"

        for V in $(ls $v_dir)
        do
                write_dir="./data/${dat}/video/${V}"
                mkdir -p $write_dir
                ffmpeg -y -r $rate -i "${v_dir}/${V}/%d.png" -r 30 \
                        ${write_dir}/${V}_tempered.mp4

                ffmpeg -y -r $rate -i "${gt_dir}/${V}/%d.jpg" -r 30 \
                        ${write_dir}/${V}_tempered_gt.mp4
        done

done
