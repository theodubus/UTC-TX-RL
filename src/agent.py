import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

MAX_ACTION_TOTAL = 1000


def get(dico, key):
    """
    Returns the value associated to the key in the dictionary, 0 otherwise.
    Prevents from storing a value for each state-action pair in the Q-table,
    even the ones that are never visited (thus saving memory).
    """
    if key in dico:
        return dico[key]
    else:
        return 0


def maxAction(Q, state, actions):
    """
    Returns the action with the highest Q value for the given state.

    Args:
    - Q (dict): the Q-table
    - state (tuple): the current state of the environment
    - actions (list): the list of possible actions

    Returns:
    - str: the action with the highest Q value
    """
    key = state[0], state[1], state[2], state[3], state[4], state[5], state[6]
    values = np.array([get(Q, key + tuple([a])) for a in actions])
    action = np.argmax(values)
    return actions[action]


def epsilonGreedy(epsilon, Q, state, actions):
    """
    Returns the action to perform according to the epsilon-greedy policy.

    Args:
    - epsilon (float): the probability of choosing a random action
    - Q (dict): the Q-table
    - state (tuple): the current state
    - actions (list): the list of possible actions

    Returns:
    - str: the action to perform
    """
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
        return maxAction(Q, state, actions)


def record(env, Q, output_file):
    """
    Enregistre sous forme de GIF l'agent evoluant dans l'environnement.
    """
    cmap = ListedColormap(['white', 'black', 'yellow', 'brown', 'orange', 'grey', 'blue'])
    observation = env.reset()
    images = []
    done = False
    nb_steps = 0

    while not done and nb_steps < MAX_ACTION_TOTAL:
        nb_steps += 1
        action = maxAction(Q, observation, env.possible_actions)
        observation, reward, done = env.step(action, render_if_illegal=True)
        fig, ax = plt.subplots()
        ax.imshow(env.grid, cmap=cmap, interpolation='none', origin='upper')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        images.append(image)
        plt.close(fig)

    if nb_steps == MAX_ACTION_TOTAL:
        print("L'agent n'a pas reussi a atteindre la fin du niveau")
    else:
        print("L'agent a atteint la fin du niveau en", nb_steps, "etapes")
        try:
            imageio.mimsave(output_file, images)
        except RuntimeError:
            print("Impossible de sauvegarder le GIF")
