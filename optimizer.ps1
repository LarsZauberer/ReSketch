python src_physics/nn_optimizer.py -r 7 -i 14 --speed
mv logs/log.log logs/log-21i-physics-speed-mnist-d.log
mkdir pretrained_models/physics-speed
cp -r src/nn_memory pretrained_models/physics-speed/

python src_physics/nn_optimizer.py -r 7 -i 14 --mnist -c datarec
mv logs/log.log logs/log-21i-physics-mnist-mnist-d.log
mkdir pretrained_models/physics-mnist
cp -r src/nn_memory pretrained_models/physics-mnist/

python src_physics/nn_optimizer.py -r 7 -i 14 --mnist --speed -c datarec
mv logs/log.log logs/log-21i-physics-mnist-speed-mnist-d.log
mkdir pretrained_models/physics-mnist-speed
cp -r src/nn_memory pretrained_models/physics-mnist-speed/
