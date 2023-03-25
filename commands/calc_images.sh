python src/test.py -g -n g.sm.0 -sm -d mnist_test -gm 0 -t 26
python src/test.py -g -n g.sm.2 -sm -d mnist_test -gm 2 -t 26
python src/test.py -g -n g.sm.8 -sm -d mnist_test -gm 8 -t 26
python src/test.py -g -n g.sm.F -sm -d emnist_test -gm 5 -t 26
python src/test.py -g -n g.sm.Flower -sm -d quickdraw_test -gm 9 -t 26

python src/test.py -g -n g.np.0 -np -d mnist_test -gm 0 -t 26
python src/test.py -g -n g.np.2 -np -d mnist_test -gm 2 -t 26
python src/test.py -g -n g.np.8 -np -d mnist_test -gm 8 -t 26
python src/test.py -g -n g.np.F -np -d emnist_test -gm 5 -t 26
python src/test.py -g -n g.np.Flower -np -d quickdraw_test -gm 9 -t 26