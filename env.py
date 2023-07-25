from functools import reduce

import gymnasium as gymnasium
import numpy as np
from gymnasium import spaces

from render_engine import render_engine
import copy

GAME_SIZE = [6, 7]
BLUE = 1
GREEN = -1

WINS_REWARD = 1000
THREE_REWARD = 3
TWO_REWARD = 1


def ingame(cord):
    return (GAME_SIZE[0] > cord[0]) and (GAME_SIZE[1] > cord[1])


class PlayBoard(gymnasium.Env):
    action_space = spaces.Discrete(GAME_SIZE[1])
    observation_space = spaces.Box(low=-1, high=1, shape=GAME_SIZE)
    reward_range = (float("-inf"), float("inf"))

    # ----------------gym api-------------------
    def __init__(self):
        self._blue_wins = 0
        self._green_wins = 0
        self._games = 0

        self._total_reward_blue = 0
        self._total_reward_green = 0

        self._game_array = np.zeros(GAME_SIZE)
        self._steps = 0
        self._maxsteps = GAME_SIZE[0] * GAME_SIZE[1]

        self.render_engine_instance = render_engine()

    # actions.lenght == GAME_SIZE[0]
    # actions probabilty array sums to 1
    # gymnasium.Env.step(self, action: ActType) → tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
    def step(self, actions: list):
        self._place(actions)
        self._steps += 1
        terminated = self._steps == self._maxsteps
        truncated = terminated
        return [copy.deepcopy(self._game_array), self._get_reward(), terminated, truncated, {}]

    def reset(self, seed=None, options=None):
        self._game_array = np.zeros(GAME_SIZE)
        return copy.deepcopy(self._game_array), {}

    def render(self):
        self.render_engine_instance.show(self._game_array)

    def close(self):
        pass

    # ----------------helpers-------------------
    def _place(self, actions: list):
        actions_sorted = actions.copy()
        actions_sorted.sort(reverse=True)
        placed = False
        for action_prob in actions_sorted:
            action_collum = actions.index(action_prob)
            for Y_cord in range(len(self._game_array) - 1, -1, -1):
                if self._game_array[Y_cord][action_collum] == 0:
                    self._game_array[Y_cord][action_collum] = self._get_current_player()
                    placed = True
                    break

            if placed:
                break

    def _get_reward(self):
        reward = 0
        attemps = self._reward_helper()
        if (wins := len(list(filter(lambda x: x[0] == 4, attemps)))) > 0:
            reward += WINS_REWARD * wins  # meerder keer 4 op een rijen is beter
            reward -= self._steps  # hoe sneller hoe beter
            return reward
        reward += sum(map(lambda x: THREE_REWARD if x[0] == 3 else TWO_REWARD, attemps))  # enkel 3,2,4 word gereward
        return reward  # indien er een 4 inzit word deze code nooit uitgevoerd

    def _reward_helper(self):
        cords = np.array(np.where(self._game_array == self._get_current_player())).T
        attempts = []  # {cord: [(length, (direction)])} direction =  vector of direction
        for cord in cords:
            usefull_directions = self._is_part_other_line(cord)
            attempts += [(x, line_dir) for line_dir in usefull_directions if
                         (x := self._get_line_in_dir(cord, line_dir)) > 1]
        return attempts

    def _get_line_in_dir(self, cord, direction):
        if ingame(cord) and self._game_array[cord[0]][cord[1]] == self._get_current_player():
            return self._get_line_in_dir((cord[0] + direction[0], cord[1] + direction[1]), direction) + 1
        return 0

    def _is_part_other_line(self, cord):
        directions = [(0, -1), (1, 0), (1, 1), (1, -1)]
        new_directions = []
        for direction in directions:
            new_cord = (cord[0] + direction[0], cord[1] + direction[1])
            if (not ingame(new_cord)) or self._game_array[new_cord[0]][new_cord[1]] != self._get_current_player():
                new_directions.append((-direction[0], -direction[1]))
        return new_directions

    def _get_current_player(self):
        return BLUE if self._steps % 2 == 0 else GREEN
