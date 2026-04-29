from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from numerical_methods import build_scalar_function, build_two_var_function, NumericalMethodError


class PlotManager: # plot manager class
    def __init__(self, parent_frame): # constructor 
        self.parent_frame = parent_frame # store tkinter frame
        self.canvas = None
        self.figure = None

    def clear(self): # method to remove old plot before drawing new one
        if self.canvas is not None: # if there is a plot in the GUI,
            self.canvas.get_tk_widget().destroy() # remove visible plot
            self.canvas = None # reset canvas
        if self.figure is not None: # if there is an active Matplotlib figure stored
            plt.close(self.figure) # close it
            self.figure = None # reset

    def draw_1d(self, expr: str, xmin: float, xmax: float, root: float | None = None): # method to draw 1 variable plot
        if xmax <= xmin: # check x-axis range
            raise NumericalMethodError('x max must be greater than x min.') # raise error if range is invalid

        f = build_scalar_function(expr) # convert user's function string into callable function
        x = np.linspace(xmin, xmax, 800) # create 800 evenly spaces x-values between xmin & xmax
        y = np.array([f(val) for val in x]) # evaluate function at every x-value

        self.clear() # clear previous plot
        fig = plt.Figure(figsize=(6, 4), dpi=100) # create new matplotlib figure
        ax = fig.add_subplot(111) # add plotting area to figure
        ax.plot(x, y, label='f(x)') # plot function curve
        ax.axhline(0, linewidth=1) # draw horizontal line @ y=0
        if root is not None: # if root value provided
            ax.plot(root, f(root), marker='o', linestyle='None', label='Approx. root') # plot point at root value
        ax.set_xlabel('x') # labels
        ax.set_ylabel('f(x)')
        ax.set_title('Function Plot')
        ax.grid(True)
        ax.legend()

        self.figure = fig # store current figure to be cleared later
        self.canvas = FigureCanvasTkAgg(fig, master=self.parent_frame) # create tkinter compatible cnavas from matplotlib figure
        self.canvas.draw() # draw figure onto canvas
        self.canvas.get_tk_widget().pack(fill='both', expand=True) # place canvas into tkinter frame

    def draw_2d(self, expr: str, xmin: float, xmax: float, ymin: float, ymax: float, mode: str = 'contour'): # method to draw 2 var plot
        if xmax <= xmin or ymax <= ymin: # check limits
            raise NumericalMethodError('Plot limits must increase from min to max.')

        f = build_two_var_function(expr) # turn user expression into callable function
        x = np.linspace(xmin, xmax, 100) # 100 x-values
        y = np.linspace(ymin, ymax, 100) # 100 y-values
        X, Y = np.meshgrid(x, y) # create grid of x/y coordinate pairs
        Z = np.array(f(X, Y), dtype=float) # evaluate function over whole grid

        self.clear() # clear previous plot
        fig = plt.Figure(figsize=(6.5, 4.5), dpi=100) # create new matplotlib figure

        if mode == 'surface': # if surface plot selected
            ax = fig.add_subplot(111, projection='3d') # create 3D plotting area
            ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True) # draw 3d surface
            ax.set_zlabel('f(x, y)') # labels
            ax.set_title('Surface Plot')
        else: # if contour plot selected
            ax = fig.add_subplot(111) # create 2d plotting area
            contour = ax.contourf(X, Y, Z, levels=20) # create filled contour plot
            fig.colorbar(contour, ax=ax) # add color bar
            ax.set_title('Contour Plot') # title

        ax.set_xlabel('x') # labels
        ax.set_ylabel('y')

        self.figure = fig # store current matplotlib figure
        self.canvas = FigureCanvasTkAgg(fig, master=self.parent_frame) # create tkinter compatible canvas
        self.canvas.draw() # draw plot onto canvas
        self.canvas.get_tk_widget().pack(fill='both', expand=True) # place plot inside GUI & resize with plot frame
