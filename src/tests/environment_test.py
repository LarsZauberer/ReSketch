import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agent_modules.environment import ShapeDraw, drawline

ref_Data = pd.read_csv("C:/Users/robin/OneDrive/Desktop/Maturarbeit/Nachzeichner-KI/paper rebuild/ref_Data.csv")
ref_Data = ref_Data.drop('Unnamed: 0', axis=1)
train_ind = np.random.choice(ref_Data.shape[1], 3000)
reference = []
for i in train_ind:
    reference.append(ref_Data.iloc[i].to_numpy().reshape(28,28))



env = ShapeDraw(28,7,reference)
plt.figure(figsize=(7,7))
plt.imshow(env.canvas, interpolation='none', cmap='gray')

#print(env.canvas_patch)

g, l, rew = env.step(30)

print(rew)
#print(l[1])

plt.imshow(env.canvas, interpolation='none', cmap='gray')
plt.show()





