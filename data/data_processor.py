import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt


class Data_Processer():
    train_test_split = 0.75

    def mnist(self, grayscales=False):
        ref_data = pd.read_csv("data/reference_data/mnist.csv")

        ref_train = ref_data.head( math.floor(ref_data.shape[0] * self.train_test_split))
        ref_test = ref_data.tail( math.floor(ref_data.shape[0] * (1-self.train_test_split)))
    
        train_data = self.mnist_json_formatter(ref_train, grayscales)
        test_data = self.mnist_json_formatter(ref_test, grayscales)

        with open("data/processed_data/mnist_train.json", "w") as f:
            json.dump(train_data, f)

        with open("data/processed_data/mnist_test.json", "w") as f:
            json.dump(test_data, f)

    def mnist_json_formatter(self, ref_data, grayscales):
        pro_data = [[] for _ in range(10)]
        #10 categories
        for i in range(10):
            #filter by images of given number
            num = ref_data.loc[ref_data["label"] == i]
            num = num.drop("label", axis=1)
        
            #append all images as list
            lis = []
            for j in  range(num.shape[0]):
                img = num.iloc[j].tolist()
                #remove grayscales if wanted
                if not grayscales:
                    for k, pix in enumerate(img):
                        if pix > 0: img[k] = 1
                lis.append(img)
            
            pro_data[i] = lis
        return pro_data


    def emnist(self, grayscales=False):
        ref_data = pd.read_csv("data/reference_data/emnist.csv")
        ref_data.columns = ["i", "labels"] + ["0"] * 784

        ref_train = ref_data.head( math.floor(ref_data.shape[0] * self.train_test_split))
        ref_test = ref_data.tail( math.floor(ref_data.shape[0] * (1-self.train_test_split)))
    
        train_data = self.emnist_json_formatter(ref_train, grayscales)
        test_data = self.emnist_json_formatter(ref_test, grayscales)

        with open("data/processed_data/emnist_train.json", "w") as f:
            json.dump(train_data, f)

        with open("data/processed_data/emnist_test.json", "w") as f:
            json.dump(test_data, f)

    def emnist_json_formatter(self, ref_data, grayscales):
        pro_data = [[] for _ in range(26)]
        for i in range(26):
            #filter by images of given letter

            num = ref_data.loc[ref_data["labels"] == (i+1)]
            num = num.drop(ref_data.columns[:2], axis=1)

            #append all images as list
            lis = []
            for j in range(len(num)):
                img = num.iloc[j].tolist()

                #Flip the image
                flipped_img = [[] for _ in range(28)]
                for k, pix in enumerate(img): 
                    flipped_img[k % 28].append(pix)
                img = []
                for col in flipped_img:
                    img += col

                #remove grayscales if wanted
                if not grayscales:
                    for k, pix in enumerate(img):
                        if pix > 0: img[k] = 1
                lis.append(img)
            
            pro_data[i] = lis

        return pro_data

    
    def quickDraw(self, grayscales=False):
        paths = [
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_anvil.npy",
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_apple.npy",
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_broom.npy",
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_bucket.npy",
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_bulldozer.npy",
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_clock.npy",
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_cloud.npy",
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_computer.npy",
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_eye.npy",
                    "src/data/reference_data/quickdraw/full_numpy_bitmap_flower.npy",
                   ]

        train_data = []
        test_data = []

        for path in paths:
            data = np.load(path, encoding="latin1", allow_pickle=True)

            if not grayscales:
                grayscale_function = lambda x: (1 if x > 0 else 0)
            else:
                grayscale_function = lambda x: x


            train = list(data[:int(self.train_test_split*1000)])
            for i, img in enumerate(train):
                train[i] = [grayscale_function(i) for i in img]

         
            test = list(data[-int(self.train_test_split*1000):])

            for i, img in enumerate(train):
                test[i] = [grayscale_function(i) for i in img]   

               

            train_data.append(train)
            test_data.append(test)

        with open("data/processed_data/quickdraw_train.json", "w") as f:
            json.dump(train_data, f)

        with open("data/processed_data/quickdraw_test.json", "w") as f:
            json.dump(test_data, f)




""" if __name__ == "__main__":
    imgs = []
    with open("data/processed_data/quickdraw_train.json") as f:
        imgs = json.load(f)

    print(len(imgs[1]))

    im = np.array(imgs[3][1]).reshape(28,28)

    fig, ax = plt.subplots()

    ax.imshow(im, cmap="gray")

    plt.show()

    print(im) """

    