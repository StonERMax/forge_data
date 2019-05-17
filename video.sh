
rate=$1
if [ -z $1 ]; then
        rate=5
fi

echo Frame Rate: $rate

v_dir="./data/davis_tempered/vid"
gt_dir="./data/davis_tempered/gt_mask"

for V in $(ls $v_dir)
do
        write_dir="./data/davis_tempered/video/${V}"
        mkdir -p $write_dir
        ffmpeg -y -r $rate -i "${v_dir}/${V}/%d.jpg" -r 30 \
                ${write_dir}/${V}_tempered.mp4

        ffmpeg -y -r $rate -i "${gt_dir}/${V}/%d.png" -r 30 \
                ${write_dir}/${V}_tempered_gt.mp4
done

