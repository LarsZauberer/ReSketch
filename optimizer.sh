source env/bin/activate

python src/nn_optimizer.py -r 7 -i 14 --mnist -c datarec
date=$(date '+%Y-%m-%d %H:%M:%S')
mv logs/log.log "logs/log-21i-base-mnist-mnist-d $date.log"
mkdir pretrained_models/base-mnist
cp -r src/nn_memory pretrained_models/base-mnist/

python src/nn_optimizer.py -r 7 -i 14 --mnist --speed -c datarec
date=$(date '+%Y-%m-%d %H:%M:%S')
mv logs/log.log "logs/log-21i-base-mnist-speed-mnist-d $date.log"
mkdir pretrained_models/base-mnist-speed
cp -r src/nn_memory pretrained_models/base-mnist-speed/

python src_physics/nn_optimizer.py -r 7 -i 14
date=$(date '+%Y-%m-%d %H:%M:%S')
mv logs/log.log "logs/log-21i-physics-base-mnist-d $date.log"
mkdir pretrained_models/physics-base
cp -r src/nn_memory pretrained_models/physics-base/
