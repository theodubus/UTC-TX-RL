import os
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Env_level(object):
    def __init__(self, file_path, render=0):
        """
        file_path (str): path to the file containing the level
        render (int): 0 if no rendering, n (> 0) if rendering with n seconds between each step
        """
        self.name_level = ""
        self.width = 0
        self.height = 0
        self.state = [0, 0, False, False, False, False, False]
        self.terminated = False
        self.file_path = file_path
        self.render_env = render
        self.possible_actions = ["U", "D", "L", "R"]
        self.grid = None
        self.initial_grid = None
        self.max_steps = 250
        self.case_association = {
            0: "empty",
            1: "wall",
            2: "key",
            3: "chest",
            4: "lava",
            5: "breakable_wall",
            6: "player"
        }
        self.cmap = None
        self.load_level(file_path)
        self.rewards = {
            "wall": -10,
            "key": 100,
            "chest": 100,
            "lava": -100,
            "out_bounds": -10,
            "action": -1,
            "break_a_wall": -5
        }

        if self.render_env > 0:
            self.cmap = ListedColormap(['white', 'black', 'yellow', 'brown', 'orange', 'grey', 'blue'])
            self.fig, self.ax = plt.subplots()
            self.img = self.ax.imshow(self.grid, cmap=self.cmap, interpolation='none', origin='upper')
            plt.xticks([])
            plt.yticks([])
            plt.show(block=False)
            self.render()

    def load_level(self, file_path):
        """
        Load the level from a file and store it in the grid attribute.
        The file must have the following format:
        - First line: name of the level
        - Second line: height and width of the grid
        - Next lines: the grid itself, with the following symbols:
            - #: wall
            - K: key
            - C: chest
            - L: lava
            - P: player
            - .: empty space
            - B: breakable wall

        Args:
        - file_path (str): path to the file containing the level

        Raises:
        - ValueError: if the file path is None or empty
        - ValueError: if the file is empty
        - ValueError: if the file does not exist

        Returns:
        - None
        """
        if file_path is None or file_path == "":
            raise ValueError("No file path provided")

        if not os.path.exists(file_path):
            raise ValueError("The file does not exist")

        if self.initial_grid is None:
            key_defined = False
            chest_defined = False
            player_defined = False
            with open(file_path, "r") as file:
                lines = file.readlines()
                if len(lines) == 0:
                    raise ValueError("The file is empty")

                # Premiere ligne : nom du niveau
                # Reste des lignes : grille du niveau
                self.name_level = lines.pop(0).strip()
                self.height = len(lines)
                self.width = len(lines[0].strip())
                self.grid = np.zeros((self.height, self.width))

                for i in range(len(lines)):
                    for j in range(len(lines[i].strip())):
                        actual = lines[i][j].upper()
                        if actual == "#":
                            self.grid[i][j] = 1

                        elif actual == "K":
                            if key_defined:
                                raise ValueError("The key is already defined")
                            self.grid[i][j] = 2
                            key_defined = True

                        elif actual == "C":
                            if chest_defined:
                                raise ValueError("The chest is already defined")
                            self.grid[i][j] = 3
                            chest_defined = True

                        elif actual == "L":
                            self.grid[i][j] = 4

                        elif actual == "P":
                            if player_defined:
                                raise ValueError("The player is already defined")
                            self.grid[i][j] = 6
                            player_defined = True

                        elif actual == "B":
                            self.grid[i][j] = 5
                        else:
                            self.grid[i][j] = 0

            if not key_defined or not chest_defined or not player_defined:
                raise ValueError("Key, chest and player must be defined")

            self.initial_grid = self.grid.copy()
        else:
            self.grid = self.initial_grid.copy()

        position = np.where(self.grid == 6)
        self.state = [position[0][0], position[1][0], False]
        self.setState(self.state)
        self.terminated = False
        self.max_steps = 250

    def setState(self, s):
        """
        Set the state of the environment to the given state by updating the position of the player and the grid.

        Args:
        - s (tuple): [pos_x, pos_y, has_key]
        - self.state: [pos_x, pos_y, has_key, breakable_wall_up, breakable_wall_down, breakable_wall_left, breakable_wall_right]

        Returns:
        - None
        """
        s.extend([False, False, False, False])

        if s[0] < 0 or s[0] >= self.height or s[1] < 0 or s[1] >= self.width:
            raise ValueError("Out of bounds")
        if self.grid[s[0], s[1]] not in [0, 2, 3, 6]:
            raise ValueError("Cannot move on lava or wall")
        if s[2] not in [True, False]:
            raise ValueError("Invalid key state")
        if self.grid[s[0], s[1]] == 3 and not s[2]:
            raise ValueError("The player needs the key to open the chest")

        up_wall = [s[0] - 1, s[1]]
        down_wall = [s[0] + 1, s[1]]
        left_wall = [s[0], s[1] - 1]
        right_wall = [s[0], s[1] + 1]

        if up_wall[0] >= 0 and self.grid[up_wall[0], up_wall[1]] == 5:
            s[3] = True
        else:
            s[3] = False

        if down_wall[0] < self.height and self.grid[down_wall[0], down_wall[1]] == 5:
            s[4] = True
        else:
            s[4] = False

        if left_wall[1] >= 0 and self.grid[left_wall[0], left_wall[1]] == 5:
            s[5] = True
        else:
            s[5] = False

        if right_wall[1] < self.width and self.grid[right_wall[0], right_wall[1]] == 5:
            s[6] = True
        else:
            s[6] = False

        if self.grid[s[0], s[1]] == 2:
            s = (s[0], s[1], True, s[3], s[4], s[5], s[6])

        if s[0] != self.state[0] or s[1] != self.state[1]:
            self.grid[s[0], s[1]] = 6
            self.grid[self.state[0], self.state[1]] = 0

        self.state = s

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        - tuple: the initial state of the environment
        """
        self.load_level(self.file_path)
        return self.state

    def step(self, action, render_if_illegal=False):
        """
        Perform the given action in the environment.

        Args:
        - action (str): the action to perform

        Returns:
        - tuple: the new state of the environment, the reward and a bool indicating if the game is over
        """
        if action not in self.possible_actions:
            raise ValueError("Invalid action")

        reward = 0
        reward += self.rewards["action"]
        legal_move = True

        if action == "U":
            new_position = (self.state[0] - 1, self.state[1], self.state[2])
        elif action == "D":
            new_position = (self.state[0] + 1, self.state[1], self.state[2])
        elif action == "L":
            new_position = (self.state[0], self.state[1] - 1, self.state[2])
        elif action == "R":
            new_position = (self.state[0], self.state[1] + 1, self.state[2])

        if new_position[0] < 0 or new_position[0] >= self.height or new_position[1] < 0 or new_position[1] >= self.width:
            reward += self.rewards["out_bounds"]
            legal_move = False

        elif self.grid[new_position[0], new_position[1]] == 1:
            reward += self.rewards["wall"]
            legal_move = False

        elif self.grid[new_position[0], new_position[1]] == 2:
            reward += self.rewards["key"]

        elif self.grid[new_position[0], new_position[1]] == 3:
            if self.state[2]:
                reward += self.rewards["chest"]
                self.terminated = True
            else:
                reward += self.rewards["wall"]
                legal_move = False

        elif self.grid[new_position[0], new_position[1]] == 4:
            reward += self.rewards["lava"]
            self.terminated = True
            legal_move = False

        elif self.grid[new_position[0], new_position[1]] == 5:
            reward += self.rewards["break_a_wall"]
            # On ne bouge pas mais on casse le mur
            self.grid[new_position[0], new_position[1]] = 0
            new_position = tuple([self.state[0], self.state[1], new_position[2]])

        if legal_move:
            self.setState(list(new_position))

        if self.render_env > 0:
            if render_if_illegal or legal_move:
                self.render()

        self.max_steps -= 1

        if self.max_steps == 0:
            self.terminated = True

        return self.state, reward, self.terminated

    def render(self):
        """
        Display the grid of the environment.

        Returns:
        - None
        """
        if self.render_env > 0:
            self.img.set_data(self.grid)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

    def get_action_sample(self):
        return random.choice(self.possible_actions)
