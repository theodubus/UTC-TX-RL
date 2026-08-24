import os
import tkinter as tk
from PIL import ImageTk

from src.gui import QLearningInterface

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Interface d'utilisation du Q-Learning")

    if os.name == 'nt':
        root.state('zoomed')
        root.iconbitmap('rl.ico')
        root.resizable(False, False)
    else:
        root.attributes('-fullscreen', True)
        root.iconphoto(True, ImageTk.PhotoImage(file="rl.ico"))

    root.protocol("WM_DELETE_WINDOW", root.quit)

    gui = QLearningInterface(root)

    root.mainloop()
