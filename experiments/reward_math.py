import numpy as np

def stop_reward(score: float, step: int) -> float:
        ACC = 3
        SPEED = 2.5
        WEIGHT = 1

        speed_factor = 1 - (step/64)**SPEED
        accuracy_factor = score**ACC

        return accuracy_factor * speed_factor * WEIGHT


""" i = 0
accuracies = []
while i < 1:
    i += 0.05
    j = 0 
    steps = []
    while j < 64:
        j += 4
        steps.append('%.5f' % stop_reward(i, j, 1))
    accuracies.append(np.array(steps))

print(np.array(accuracies)) """



print( 0.6178343949044586 / ((1 - (19/64)*2.5) * 0.7) ) 