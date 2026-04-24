from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from numerical_methods import build_scalar_function, build_two_var_function, NumericalMethodError


class PlotManager:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.canvas = None
        self.figure = None

    def clear(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None

    def draw_1d(self, expr: str, xmin: float, xmax: float, root: float | None = None):
        if xmax <= xmin:
            raise NumericalMethodError('x max must be greater than x min.')

        f = build_scalar_function(expr)
        x = np.linspace(xmin, xmax, 800)
        y = np.array([f(val) for val in x])

        self.clear()
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, y, label='f(x)')
        ax.axhline(0, linewidth=1)
        if root is not None:
            ax.plot(root, f(root), marker='o', linestyle='None', label='Approx. root')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Function Plot')
        ax.grid(True)
        ax.legend()

        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def draw_2d(self, expr: str, xmin: float, xmax: float, ymin: float, ymax: float, mode: str = 'contour'):
        if xmax <= xmin or ymax <= ymin:
            raise NumericalMethodError('Plot limits must increase from min to max.')

        f = build_two_var_function(expr)
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array(f(X, Y), dtype=float)

        self.clear()
        fig = plt.Figure(figsize=(6.5, 4.5), dpi=100)

        if mode == 'surface':
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
            ax.set_zlabel('f(x, y)')
            ax.set_title('Surface Plot')
        else:
            ax = fig.add_subplot(111)
            contour = ax.contourf(X, Y, Z, levels=20)
            fig.colorbar(contour, ax=ax)
            ax.set_title('Contour Plot')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
