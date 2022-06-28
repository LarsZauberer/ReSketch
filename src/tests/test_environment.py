import math

def action_to_direction(action):
    counter = 0
    total_counter = 1
    for i in range(action):
        counter += 1
        if counter >= 4:
            counter = 0
            total_counter += 1
    
    if counter == 0:
        x = 0
        y = 1 * total_counter
    elif counter == 1:
        x = 1 * total_counter
        y = 0
    elif counter == 2:
        x = 0
        y = -1 * total_counter
    elif counter == 3:
        x = -1 * total_counter
        y = 0
        
    if action // 4 >= 1:
        isDrawing = 0
        if x != 0:
            x -= math.copysign(1, x)
        if y != 0:
            y -= math.copysign(1, y)
    else:
        isDrawing = 1
    
    return x, y, isDrawing


def test_action_1():
    assert action_to_direction(0) == (0, 1, 1)


def test_action_2():
    assert action_to_direction(1) == (1, 0, 1)
    

def test_action_3():
    assert action_to_direction(2) == (0, -1, 1)


def test_action_4():
    assert action_to_direction(3) == (-1, 0, 1)


def test_action_5():
    assert action_to_direction(4) == (0, 1, 0)


def test_action_6():
    assert action_to_direction(5) == (1, 0, 0)
    

def test_action_7():
    assert action_to_direction(6) == (0, -1, 0)


def test_action_8():
    assert action_to_direction(7) == (-1, 0, 0)
        