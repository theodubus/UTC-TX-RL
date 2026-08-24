import os
import queue
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from src.env import Env_level
from src.agent import maxAction, epsilonGreedy, get, record, MAX_ACTION_TOTAL


def get_image_level(level_file):
    """
    Returns the image of the level as a numpy array.
    """
    cmap = ListedColormap(['white', 'black', 'yellow', 'brown', 'orange', 'grey', 'blue'])
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

        # Frame de gauche pour les entrees utilisateur
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side='left', fill='both', expand=True)

        input_label = ttk.Label(left_frame, text="Veuillez saisir les informations pour l'agent", font=("Helvetica", 12))
        input_label.pack(pady=10)

        # Saisie du fichier de niveau
        file_entry = ttk.Entry(left_frame, textvariable=self.file_var, width=50)
        file_entry.pack(pady=5)
        file_button = ttk.Button(left_frame, text="Choisir le fichier", command=self.load_file)
        file_button.pack(pady=5)

        # Choix du nombre d'iterations
        iteration_label = ttk.Label(left_frame, text="Nombre d'iterations:")
        iteration_label.pack(pady=5)
        iteration_slider = tk.Scale(left_frame, from_=1000, to=20000, orient='horizontal', variable=self.iteration_var, resolution=100)
        iteration_slider.pack(pady=5)

        self.start_button = ttk.Button(left_frame, text="Debut apprentissage", command=self.start_learning, width=30, state='disabled')
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

        # Frame de droite pour les affichages (recompenses et gif)
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side='right', fill='both', expand=True)

        # Sous-frame pour les recompenses
        self.rewards_frame = ttk.Frame(right_frame)
        self.rewards_frame.place(x=0, y=0, height=quarter_height)

        self.fig = plt.Figure(dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylabel("Recompense totale")
        self.fig.suptitle("Evolution des recompenses au cours de l'apprentissage")
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
        if self.queue_animation.qsize() == 0:
            return

        action = self.queue_animation.get()
        if action["type"] == "reward_plot":
            self.ax.clear()
            self.ax.set_ylabel("Recompense totale")
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
            tk.messagebox.showerror("Erreur", "Veuillez choisir un nombre d'iterations")
            return

        self.start_button.config(state='disabled')

        self.ax.clear()
        self.ax.set_ylabel("Recompense totale")
        self.canvas.draw()
        plt.close('all')

        self.restart_button.config(state='disabled')
        self.gif_label.config(image=None)
        self.gif_frames = []
        self.gif_label.config(image=None)

        new_thread = threading.Thread(target=self.learn_and_update, args=(level_file, num_iterations,), daemon=True)
        new_thread.start()

        return

    def learn_and_update(self, level_file, num_iterations):
        if os.path.exists("outputs/agent.gif"):
            os.remove("outputs/agent.gif")

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

            if i % 1000 == 0:
                self.queue_animation.put({"type": "reward_plot", "x_data": np.arange(1, i + 1), "y_data": totalRewards[:i]})
                self.event_generate("<<UpdatePlot>>", when="tail")

        self.queue_animation.put({"type": "reward_plot", "x_data": np.arange(1, num_iterations + 1), "y_data": totalRewards})
        self.event_generate("<<UpdatePlot>>", when="tail")

        os.makedirs("outputs", exist_ok=True)
        self.queue_animation.put({"type": "gif", "env": env, "Q": Q, "gif_path": "outputs/agent.gif"})
        self.event_generate("<<UpdatePlot>>", when="tail")
