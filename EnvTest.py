import random
import time
import numpy as np
from env import PlayBoard

def test_env():
    env = PlayBoard()

    while True:
        actions = np.random.random(env.action_space.n)
        actions /= actions.sum()
        print(actions)

        env.step(list(actions))
        env.render()
        time.sleep(10)

def test_reward():
    env = PlayBoard()           #(0,0)
    env._game_array = np.array([[0, 0, 0, 0, 0, 0, 0], #(0,6)
                                [0, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0]])#(6,6)
    assert env._reward_helper() == [(4, (-1, 0))] # up
    assert env._get_reward() == 1000

    env = PlayBoard()
    env._game_array = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0]])
    assert env._reward_helper() == [(4, (0, 1))] # to right
    assert env._get_reward() == 1000

    env = PlayBoard()
    env._game_array = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0]])
    assert env._reward_helper() == [(4, (-1, 1))] # schuin up
    assert env._get_reward() == 1000

    env = PlayBoard()
    env._game_array = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, -1, 0, 0, 0, 0],
                                [0, 1, -1, 1, 1, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0]])
    assert env._reward_helper().count((2, (-1, 1))) == 1
    assert env._reward_helper().count((2, (0, 1))) == 1
    assert len(env._reward_helper()) == 2
    assert env._get_reward() == 2

    env._steps =1
    assert env._reward_helper() == [(2, (-1, 0))]
    assert env._get_reward() == 1
    env = PlayBoard()
    env._game_array = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [1, 1, -1, 0, 0, 0, 0],
                                [1, 1, -1, 1, 1, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0]])
    assert env._get_reward() == (3+1+1+1+1+1+1+1)
    print("oke")




test_reward()


