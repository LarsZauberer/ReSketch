import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import cm, colors

def generate_image(images, columns=2):
        num = len(images)
        rows = int(num/columns)
        
        labeled = 0

        fig = plt.figure(figsize=(10., 15.))
        grid = ImageGrid(fig, 111,  
                        nrows_ncols=(rows, columns*3), 
                        axes_pad=0.1,  
                        )

        interval = np.full((28, 1), 255)
        sorted_images = []
        for index, item in enumerate(images):
            ref, canv = item
            sorted_images.append((ref.reshape((28,28)), False))
            sorted_images.append((canv.reshape((28,28)), True))
            sorted_images.append(interval.copy())

        for ax, im in zip(grid, sorted_images):
            # Iterating over the grid returns the Axes.
            ax.axis("off")
            if len(im) == 2:
                ax.imshow(im[0], cmap="bone", vmin=0, vmax=255)
                if labeled < columns*2:
                    if im[1]:
                        ax.set_title("ReSketch")
                        labeled += 1
                    else:
                        ax.set_title("Reference")
                        labeled += 1
            else:
                ax.imshow(im, cmap="bone", vmin=0, vmax=255)
        
        fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=64), cmap="bone"), ax=grid, orientation="horizontal", fraction=0.046, pad=0.04, label="Steps", location="bottom")

        plt.savefig(f"src/data_statistics/ai_images.png", bbox_inches='tight')
        plt.pause(10)


def generate_generative_image(images, columns=3):
    num = len(images)
    rows = int(num/columns)
    

    fig = plt.figure(figsize=(10., 15.))
    grid = ImageGrid(fig, 111,  
                    nrows_ncols=(rows, columns), 
                    axes_pad=0.1,  
                    )

    sorted_images = []
    for img in images:
        sorted_images.append(img.reshape((28,28)))

    for ax, im in zip(grid, sorted_images):
        # Iterating over the grid returns the Axes.
        ax.axis("off")
        ax.imshow(im, cmap="bone", vmin=0, vmax=255)
    
    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=64), cmap="bone"), ax=grid, orientation="horizontal", fraction=0.046, pad=0.5, label="Steps", location="bottom")

    plt.savefig(f"src/data_statistics/ai_images.png", bbox_inches='tight')
    plt.pause(5)



#image of all datasets
""" if args.image:
images = []
test = Test_NN(n_test=200, dataset="mnist", version=args.version)
test.test_from_loaded(agent_args=test.agent_args, mode="vis")
images1 = test.images[:8]

test = Test_NN(n_test=200, dataset="emnist", version=args.version)
test.test_from_loaded(agent_args=test.agent_args, mode="vis")
images2 = test.images[:8]

test = Test_NN(n_test=200, dataset="quickdraw", version=args.version)
test.test_from_loaded(agent_args=test.agent_args, mode="vis")
images3 = test.images[:8]

for i in range(8):
    images.append(images1[i])
    images.append(images2[i])
    images.append(images3[i])

test.images = images
test.generate_image(columns=3)
exit() """