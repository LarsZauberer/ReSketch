source env/bin/activate
python src/nn_test.py -s -t 2000 --version base
python src/nn_test.py -s -t 2000 --version mnist
python src/nn_test.py -s -t 2000 --version speed
python src/nn_test.py -s -t 2000 --version mnist-speed
python src_physics/nn_test.py -s -t 2000 --version base
python src_physics/nn_test.py -s -t 2000 --version mnist
python src_physics/nn_test.py -s -t 2000 --version speed
python src_physics/nn_test.py -s -t 2000 --version mnist-speed