#name="greeble_keras"
#path_out="/home/ec1018/greebles_inv/misc_tests/finetuned-resnet50-keras-master"
#sbatch --job-name=$name --output=$path_out/$name.o --error=$path_out/$name.e run-project-keras.sh

name2="greeble_resnet_top"
name3="greeble_vgg16"
path_out2="/home/ec1018/greebles_inv/misc_tests/resnet_include_top"
path_out3="/home/ec1018/greebles_inv/misc_tests/vgg16"

#sbatch --job-name=$name2 --output=$path_out2/$name2.o --error=$path_out2/$name2.e run-resnet_top.sh
sbatch --job-name=$name3 --output=$path_out3/$name3.o --error=$path_out3/$name3.e run-vgg16.sh