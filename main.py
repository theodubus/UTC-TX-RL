import os
import time
import queue
import random
import imageio
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

MAX_ACTION_TOTAL = 1000

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
            self.cmap = ListedColormap(['white','black','yellow','brown','orange','grey', 'blue'])
            self.fig, self.ax = plt.subplots()
            self.img = self.ax.imshow(self.grid, cmap=self.cmap, interpolation='none', origin='upper')
            plt.xticks([])
            plt.yticks([])
            plt.show(block=False) 
            self.render()


    
    def load_level(self, file_path):
        """ 
        Load the level from a file and store it in the grid attribute
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
        
        if file_path == None or file_path == "":
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
                
                # Première ligne : nom du niveau
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
            self.grid = self.initial_grid.copy() # evite de relire le fichier à chaque reset

        position = np.where(self.grid == 6)
        self.state = [position[0][0], position[1][0], False]
        self.setState(self.state)
        self.terminated = False
        self.max_steps = 250
        

    def setState(self, s):
        """
        Set the state of the environment to the given state by updating the position of the player and the grid
        
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
        Reset the environment to its initial state
        
        Returns:
        - tuple: the initial state of the environment
        """
        self.load_level(self.file_path)
        return self.state


    def step(self, action, render_if_illegal=False):
        """
        Perform the given action in the environment
        
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

            # finalement on ne bouge pas mais on casse le mur
            self.grid[new_position[0], new_position[1]] = 0
            new_position = tuple([self.state[0], self.state[1], new_position[2]])

        
        if legal_move:
            self.setState(list(new_position))

        if self.render_env > 0:
            if render_if_illegal or legal_move:
                self.render()
                # time.sleep(self.render_env)
        
        self.max_steps -= 1

        if self.max_steps == 0:
            self.terminated = True

        return self.state, reward, self.terminated
        

    def render(self): 
        """
        Display the grid of the environment
        
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


def record(env, Q, output_file):
    """
    On enregistre sous forme de GIF l'agent évoluant dans l'environnement
    """
    cmap = ListedColormap(['white','black','yellow','brown','orange','grey', 'blue'])
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
        
        # Convert plot to numpy array
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        images.append(image)
        plt.close(fig)
    if nb_steps == MAX_ACTION_TOTAL:
        print("L'agent n'a pas réussi à atteindre la fin du niveau")
    else:
        print("L'agent a atteint la fin du niveau en", nb_steps, "étapes")
        try :
            imageio.mimsave(output_file, images)
        except RuntimeError:
            print("Impossible de sauvegarder le GIF")
            


def maxAction(Q, state, actions):
    """
    Returns the action with the highest Q value for the given state
    
    Args:
    - Q (ndarray): the Q-table
    - state (tuple): the current state of the environment : (x, y, has_key)
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
    Returns the action to perform according to the epsilon-greedy policy
    
    Args:
    - epsilon (float): the probability of choosing a random action
    - Q (ndarray): the Q-table
    - state (int): the current state
    
    Returns:
    - str: the action to perform
    """
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
        return maxAction(Q, state, actions)

def get(dico, key):
    """
    Returns the value associated to the key in the dictionary, 0 otherwise
    Prevents from storing a value for each state-action pair in the Q-table, even the ones
    that are never visited (thus saving memory)
    """
    if key in dico:
        return dico[key]
    else:
        return 0
    
def get_image_level(level_file):
    """
    Returns the image of the level
    """
    cmap = ListedColormap(['white','black','yellow','brown','orange','grey', 'blue'])
    grid = []
    with open(level_file, "r") as file:
        lines = file.readlines()
        if len(lines) == 0:
            raise ValueError("The file is empty")
        
        lines.pop(0)  # Remove the name of the level
        h = len(lines)
        w = len(lines[0].strip())
        grid = np.zeros((h, w))
        
        for i in range(len(lines)):
            for j in range(len(lines[i].strip())):
                actual = lines[i][j].upper()
                if actual == "#":
                    grid[i][j] = 1
                
                elif actual == "K":
                    grid[i][j] = 2
                    
                elif actual == "C":
                    grid[i][j] = 3
                    
                elif actual == "L":
                    grid[i][j] = 4
                    
                elif actual == "P":
                    grid[i][j] = 6

                elif actual == "B":
                    grid[i][j] = 5
                else:
                    grid[i][j] = 0
                    
    # Convert the grid to an image
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=cmap, interpolation='none', origin='upper')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return img

class QLearningInterface(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        
        self.root = root
        self.queue_animation = queue.Queue()
        
        self.file_var = tk.StringVar()
        self.iteration_var = tk.IntVar(value=1000)
                
        screen_height = self.root.winfo_screenheight()
        quarter_height = screen_height // 3

        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True)

        # Frame pour le titre
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x')
        title_label = ttk.Label(title_frame, text="Interface d'utilisation du Q-Learning", font=("Helvetica", 16))
        title_label.pack(pady=10)

        # Frame pour le contenu
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True)

        # Frame de gauche pour les entrées utilisateur
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side='left', fill='both', expand=True)
        
        input_label = ttk.Label(left_frame, text="Veuillez saisir les informations pour l'agent", font=("Helvetica", 12))
        input_label.pack(pady=10)

        # Saisie du fichier de niveau
        file_entry = ttk.Entry(left_frame, textvariable=self.file_var, width=50)
        file_entry.pack(pady=5)
        file_button = ttk.Button(left_frame, text="Choisir le fichier", command=self.load_file)
        file_button.pack(pady=5)

        # Choix du nombre d'itérations avec un slider et affichage de la valeur dans un stringvar du label
        iteration_label = ttk.Label(left_frame, text="Nombre d'itérations:")
        iteration_label.pack(pady=5)
        iteration_slider = tk.Scale(left_frame, from_=1000, to=20000, orient='horizontal', variable=self.iteration_var, resolution=100)
        iteration_slider.pack(pady=5)

        self.start_button = ttk.Button(left_frame, text="Début apprentissage", command=self.start_learning, width=30, state='disabled')
        self.start_button.pack(pady=10)
        
        # Bouton pour relancer le gif
        self.restart_button = ttk.Button(left_frame, text="Relancer le GIF", command=lambda: self.update_gif(0), width=30, state='disabled')
        self.restart_button.pack(pady=10)
        
        # Bouton Quitter interface
        end_button = ttk.Button(left_frame, text="Quitter", command=self.root.quit, style='TButton', width=30)
        end_button.pack(pady=10)
        
        # Frame contenant l'image du niveau
        self.level_frame = ttk.Frame(left_frame)
        self.level_frame.pack(pady=10)
        

        # Frame de droite pour les affichages (récompenses et gif)
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side='right', fill='both', expand=True)

        # Sous-frame pour les récompenses
        self.rewards_frame = ttk.Frame(right_frame)
        self.rewards_frame.place(x=0, y=0, height=quarter_height)

        self.fig = plt.Figure(dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylabel("Récompense totale")
        self.fig.suptitle("Evolution des récompenses au cours de l'apprentissage")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.rewards_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Sous-frame pour le gif
        gif_frame = ttk.Frame(right_frame)
        gif_frame.place(x=0, y=quarter_height + 10)

        self.gif_label = ttk.Label(gif_frame, image=None)
        self.gif_label.pack(side='right', fill='both', expand=True)
        
        self.bind("<<UpdatePlot>>", self.process_queue)
                
    def visualize_level(self):
        if self.file_var.get():
            img = get_image_level(self.file_var.get())
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            for widget in self.level_frame.winfo_children():
                widget.destroy()
            
            level_label = ttk.Label(self.level_frame, image=img)
            level_label.image = img
            level_label.pack()

        else:
            tk.messagebox.showerror("Erreur", "Veuillez choisir un fichier de niveau")

    def load_file(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        if not os.path.exists(file_path):
            tk.messagebox.showerror("Erreur", "Le fichier n'existe pas")
            return
        
        self.file_var.set(file_path)
        self.visualize_level()
        self.start_button.config(state='normal')
        
    def process_queue(self, event):
        # Process queue
        # Si vide on ne fait rien
        if self.queue_animation.qsize() == 0:
            return
        
        action = self.queue_animation.get()
        if action["type"] == "reward_plot":
            self.ax.clear()
            self.ax.set_ylabel("Récompense totale")
            self.ax.plot(action["x_data"], action["y_data"])
            self.canvas.draw()
            
        elif action["type"] == "gif":
            record(action["env"], action["Q"], action["gif_path"])
            self.display_gif(action["gif_path"])
        
    def display_gif(self, gif_path):
        self.restart_button.config(state='normal')
        gif = imageio.mimread(gif_path)
        self.gif_frames = [tk.PhotoImage(data=imageio.imwrite(imageio.RETURN_BYTES, frame, format='gif')) for frame in gif]
        self.gif_label.config(image=None)
        
        if len(self.gif_frames) > 0:
            self.update_gif(0)

    def update_gif(self, frame_idx):
        if frame_idx == 0:
            self.restart_button.config(state='disabled')
            self.start_button.config(state='disabled')
        if frame_idx < len(self.gif_frames):
            self.gif_label.config(image=self.gif_frames[frame_idx])
            self.root.after(100, self.update_gif, frame_idx + 1)
        elif frame_idx == len(self.gif_frames):
            self.restart_button.config(state='normal')
            self.start_button.config(state='normal')
            self.gif_label.config(image=None)


    def start_learning(self):
        level_file = self.file_var.get()
        num_iterations = self.iteration_var.get()
        
        if not level_file:
            tk.messagebox.showerror("Erreur", "Veuillez choisir un fichier de niveau")
            return
        
        if not num_iterations:
            tk.messagebox.showerror("Erreur", "Veuillez choisir un nombre d'itérations")
            return
        
        self.start_button.config(state='disabled')
        
        # Clear the plot and close all figures
        self.ax.clear()
        self.ax.set_ylabel("Récompense totale")
        self.canvas.draw()
        plt.close('all')  # Close all matplotlib figures
        
        # Empty the GIF label
        self.restart_button.config(state='disabled')
        self.gif_label.config(image=None)
        self.gif_frames = []

        self.gif_label.config(image=None)
        
        new_thread = threading.Thread(target=self.learn_and_update, args=(level_file, num_iterations, ), daemon=True)
        new_thread.start()
        
        return

    def learn_and_update(self, level_file, num_iterations):
        # Delete existing agent.gif file if it exists
        if os.path.exists("agent.gif"):
            os.remove("agent.gif")
            
        env = Env_level(level_file, render=0)

        ALPHA = 0.1
        GAMMA = 1
        EPS = 1.0

        Q = {}

        totalRewards = np.zeros(num_iterations)

        for i in range(1, num_iterations + 1):
            done = False
            epRewards = 0
            observation = env.reset()
            nb_steps = 0

            while not done and nb_steps < MAX_ACTION_TOTAL:
                nb_steps += 1
                action = epsilonGreedy(EPS, Q, observation, env.possible_actions)
                observation_, reward, done = env.step(action)
                epRewards += reward
                action_ = maxAction(Q, observation_, env.possible_actions)
                state = tuple(list(observation) + [action])
                oldQ = get(Q, state)
                Q[state] = oldQ + ALPHA * (reward + GAMMA * get(Q, tuple(list(observation_) + [action_])) - oldQ)
                observation = observation_

            EPS -= 2 / num_iterations if EPS > 0 else 0
            totalRewards[i - 1] = epRewards

            # Update the plot every 1000 iterations
            if i % 1000 == 0:
                self.queue_animation.put({"type": "reward_plot", "x_data": np.arange(1, i + 1), "y_data": totalRewards[:i]})
                self.event_generate("<<UpdatePlot>>", when="tail")
                
                
        self.queue_animation.put({"type": "reward_plot", "x_data": np.arange(1, num_iterations + 1), "y_data": totalRewards})
        self.event_generate("<<UpdatePlot>>", when="tail")

        self.queue_animation.put({"type": "gif", "env": env, "Q": Q, "gif_path": "agent.gif"})
        self.event_generate("<<UpdatePlot>>", when="tail")
        

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Interface d'utilisation du Q-Learning")
    
    if os.name == 'nt':
        root.state('zoomed')
        root.iconbitmap('rl.ico')
        root.resizable(False, False)
    else:
        root.attributes('-fullscreen', True)  # Plein écran
        root.iconphoto(True, ImageTk.PhotoImage(file="rl.ico"))
        
    root.protocol("WM_DELETE_WINDOW", root.quit)
    
    gui = QLearningInterface(root)
    
    root.mainloop()