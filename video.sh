


ffmpeg -y -i "./data/davis_tempered/vid/0/%d.jpg" -framerate 5 \
        ./data/davis_tempered/video/0_tempered.mp4

ffmpeg -y -i "./data/davis_tempered/gt_mask/0/%d.png" -framerate 5 \
        ./data/davis_tempered/video/0_tempered_gt.mp4