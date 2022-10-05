source env/bin/activate
python src/nn_test.py -s -t 2000 --version base
python src/nn_test.py -s -t 2000 --version mnist
python src/nn_test.py -s -t 2000 --version speed
python src/nn_test.py -s -t 2000 --version mnist-speed
python src_physics/nn_test.py -s -t 2000 --version base
python src_physics/nn_test.py -s -t 2000 --version mnist
python src_physics/nn_test.py -s -t 2000 --version speed
python src_physics/nn_test.py -s -t 2000 --version mnist-speed

python src/nn_test.py -s -t 2000 --version base -d emnist
python src/nn_test.py -s -t 2000 --version mnist -d emnist
python src/nn_test.py -s -t 2000 --version speed -d emnist
python src/nn_test.py -s -t 2000 --version mnist-speed -d emnist
python src_physics/nn_test.py -s -t 2000 --version base -d emnist
python src_physics/nn_test.py -s -t 2000 --version mnist -d emnist
python src_physics/nn_test.py -s -t 2000 --version speed -d emnist
python src_physics/nn_test.py -s -t 2000 --version mnist-speed -d emnist

python src/nn_test.py -s -t 2000 --version base -d quickdraw
python src/nn_test.py -s -t 2000 --version mnist -d quickdraw
python src/nn_test.py -s -t 2000 --version speed -d quickdraw
python src/nn_test.py -s -t 2000 --version mnist-speed -d quickdraw
python src_physics/nn_test.py -s -t 2000 --version base -d quickdraw
python src_physics/nn_test.py -s -t 2000 --version mnist -d quickdraw
python src_physics/nn_test.py -s -t 2000 --version speed -d quickdraw
python src_physics/nn_test.py -s -t 2000 --version mnist-speed -d quickdraw