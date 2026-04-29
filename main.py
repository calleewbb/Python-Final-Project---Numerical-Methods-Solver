import tkinter as tk
from tkinter import messagebox, ttk

from numerical_methods import ( # import methods
    NumericalMethodError,
    bisection_method,
    derivative_approx,
    format_matrix,
    linear_algebra_operations,
    newton_method,
    newton_system_method,
    secant_method,
)
from plot_utils import PlotManager # import plot manager


class NumericalMethodsGUI: # main GUI class
    def __init__(self, root): # constructor 
        self.root = root # main tkinter window
        self.root.title('Numerical Methods Solver and Visualization Tool') # set title
        self.root.geometry('1200x760') # set window size

        container = ttk.Panedwindow(root, orient='horizontal') # create split screen (left/right) panels
        container.pack(fill='both', expand=True) # fill entire window

        self.left_panel = ttk.Frame(container, padding=10) # inputs on left
        self.right_panel = ttk.Frame(container, padding=10) # outputs & plots on right
        container.add(self.left_panel, weight=3) # make left side bigger (3:2 ratio)
        container.add(self.right_panel, weight=2)

        self.notebook = ttk.Notebook(self.left_panel) # create tabbed interface (RF, Diff, LA, NS, 2-Var)
        self.notebook.pack(fill='both', expand=True)

        self.output = tk.Text(self.right_panel, wrap='word', width=45, font=('Consolas', 10)) # text area for results
        self.output.pack(fill='x', expand=False)

        plot_frame = ttk.LabelFrame(self.right_panel, text='Plot Area', padding=8) # create box for plots
        plot_frame.pack(fill='both', expand=True, pady=(10, 0))
        self.plot_manager = PlotManager(plot_frame) # draw plots

        self._build_root_tab() # build tabs for each method with inputs & buttons
        self._build_derivative_tab()
        self._build_linear_algebra_tab()
        self._build_system_tab()
        self._build_plot_tab()

    def write_output(self, text: str): 
        self.output.delete('1.0', tk.END) # clear previous output
        self.output.insert(tk.END, text) # write new output

    def append_output(self, text: str):
        self.output.insert(tk.END, text) # add text without clearing

    def _safe_call(self, func): # safe execution wrapper
        try:
            func() # run whatever function passed in
        except NumericalMethodError as exc: # handle errors
            messagebox.showerror('Numerical Method Error', str(exc))
        except Exception as exc:
            messagebox.showerror('Error', str(exc))

    def _build_root_tab(self): # create root finding tab --------------------------------------------------------------------------
        frame = ttk.Frame(self.notebook, padding=12) # create tab 
        self.notebook.add(frame, text='Root Finding') 

        self.root_expr = tk.StringVar(value='x^3 - x - 2') # tkinter variable linked to input box & default value
        self.root_method = tk.StringVar(value='Bisection') # store user input for method
        self.root_a = tk.StringVar(value='1') # endpoints
        self.root_b = tk.StringVar(value='2')
        self.root_x0 = tk.StringVar(value='1.5') # initial guesses
        self.root_x1 = tk.StringVar(value='2.0')
        self.root_tol = tk.StringVar(value='1e-6') # tolerance
        self.root_iter = tk.StringVar(value='50') # iterations
        self.root_xmin = tk.StringVar(value='-5') # range
        self.root_xmax = tk.StringVar(value='5')

        fields = [ # field definitions (label, variable)
            ('f(x):', self.root_expr),
            ('Method:', self.root_method),
            ('a:', self.root_a),
            ('b:', self.root_b),
            ('x0:', self.root_x0),
            ('x1:', self.root_x1),
            ('Tolerance:', self.root_tol),
            ('Max iterations:', self.root_iter),
            ('Plot x min:', self.root_xmin),
            ('Plot x max:', self.root_xmax),
        ]

        for r, (label, var) in enumerate(fields): # loop to create UI, places inputs vertically
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky='w', padx=5, pady=4) # label
            if label == 'Method:': # special case
                combo = ttk.Combobox(frame, textvariable=var, state='readonly', # drop down instead of text box
                                     values=['Bisection', 'Newton-Raphson', 'Secant']) # options
                combo.grid(row=r, column=1, sticky='ew', padx=5, pady=4) 
            else: # text box otherwise
                ttk.Entry(frame, textvariable=var, width=28).grid(row=r, column=1, sticky='ew', padx=5, pady=4)

        frame.columnconfigure(1, weight=1) # allow input boxes to stretch if needed

        # create buttons to solve & plot
        ttk.Button(frame, text='Solve', command=lambda: self._safe_call(self.solve_root)).grid(row=10, column=0, pady=12)
        ttk.Button(frame, text='Plot f(x)', command=lambda: self._safe_call(self.plot_root_function)).grid(row=10, column=1, pady=12, sticky='w')

    def solve_root(self): # root solving
        expr = self.root_expr.get() # get user input from GUI
        method = self.root_method.get() 
        tol = float(self.root_tol.get()) # convert to numbers
        max_iter = int(self.root_iter.get())

        if method == 'Bisection': # call bisection method
            root, iters, history = bisection_method(expr, float(self.root_a.get()), float(self.root_b.get()), tol, max_iter)
        elif method == 'Newton-Raphson': # call newton
            root, iters, history = newton_method(expr, float(self.root_x0.get()), tol, max_iter)
        else: # call secant
            root, iters, history = secant_method(expr, float(self.root_x0.get()), float(self.root_x1.get()), tol, max_iter)

        lines = [ # result formatting
            f'Method: {method}',
            f'Approximate root: {root:.10f}',
            f'Iterations used: {iters}',
            '',
            'Iteration history:',
        ]
        for row in history[:12]: # only show first 12 iterations to avoid clutter
            lines.append(str(row))
        if len(history) > 12:
            lines.append('...') # add ... to end of 12th iteration if more iterations

        self.write_output('\n'.join(lines)) # display results in output box
        self.plot_manager.draw_1d(expr, float(self.root_xmin.get()), float(self.root_xmax.get()), root=root) # plot function & root

    def plot_root_function(self): # plot only, no solving results
        self.plot_manager.draw_1d(self.root_expr.get(), float(self.root_xmin.get()), float(self.root_xmax.get()))
        self.write_output('Plotted f(x).')

    def _build_derivative_tab(self): # differentiation --------------------------------------------------------------------------
        frame = ttk.Frame(self.notebook, padding=12) # create tab
        self.notebook.add(frame, text='Differentiation')

        self.diff_expr = tk.StringVar(value='sin(x) + x^2') # tkinter variable linked to input box & default value
        self.diff_x0 = tk.StringVar(value='1.0') # point where deriv is evaluated
        self.diff_h = tk.StringVar(value='0.001') # step size
        self.diff_xmin = tk.StringVar(value='-5') # range
        self.diff_xmax = tk.StringVar(value='5')

        fields = [ # field definitions
            ('f(x):', self.diff_expr),
            ('x0:', self.diff_x0),
            ('h:', self.diff_h),
            ('Plot x min:', self.diff_xmin),
            ('Plot x max:', self.diff_xmax),
        ]
        for r, (label, var) in enumerate(fields): # create input boxes
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky='w', padx=5, pady=4)
            ttk.Entry(frame, textvariable=var, width=28).grid(row=r, column=1, sticky='ew', padx=5, pady=4)

        frame.columnconfigure(1, weight=1) # allow box to stretch

        # differentiation & plot buttons
        ttk.Button(frame, text='Approximate Derivative', command=lambda: self._safe_call(self.solve_derivative)).grid(row=5, column=0, pady=12)
        ttk.Button(frame, text='Plot f(x)', command=lambda: self._safe_call(self.plot_derivative_function)).grid(row=5, column=1, pady=12, sticky='w')

    def solve_derivative(self): # derivative solving
        # call derivative_approx
        results = derivative_approx(self.diff_expr.get(), float(self.diff_x0.get()), float(self.diff_h.get()))
        lines = [ # result formatting
            'Derivative approximation results:',
            f"f(x0) = {results['f(x0)']:.10f}",
            f"Forward difference  = {results['forward']:.10f}",
            f"Backward difference = {results['backward']:.10f}",
            f"Central difference  = {results['central']:.10f}",
        ]
        self.write_output('\n'.join(lines)) # display results in output box
        self.plot_manager.draw_1d(self.diff_expr.get(), float(self.diff_xmin.get()), float(self.diff_xmax.get())) # plot f(x)

    def plot_derivative_function(self): # plot only
        self.plot_manager.draw_1d(self.diff_expr.get(), float(self.diff_xmin.get()), float(self.diff_xmax.get()))
        self.write_output('Plotted f(x) for the differentiation tab.')

    def _build_linear_algebra_tab(self): # linear algebra --------------------------------------------------------------------------
        frame = ttk.Frame(self.notebook, padding=12) # create tab
        self.notebook.add(frame, text='Linear Algebra')

        self.la_matrix = tk.StringVar(value='1 2; 3 4') # tkinter variable with default value

        # create input box
        ttk.Label(frame, text='Matrix A (2x2 or 3x3):').grid(row=0, column=0, sticky='w', padx=5, pady=4)
        ttk.Entry(frame, textvariable=self.la_matrix, width=36).grid(row=0, column=1, sticky='ew', padx=5, pady=4)

        instructions = ( # tell user how to entry matrix
            'Enter rows separated by semicolons.\n'
            'Example 2x2: 1 2; 3 4\n'
            'Example 3x3: 1 2 3; 4 5 6; 7 8 10'
        )
        # display instructions
        ttk.Label(frame, text=instructions, justify='left').grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=4)

        frame.columnconfigure(1, weight=1) # allow stretch
        # solve button
        ttk.Button(frame, text='Compute Matrix Properties', command=lambda: self._safe_call(self.solve_linear_algebra)).grid(row=2, column=0, pady=12)

    def solve_linear_algebra(self): # matrix solving
        results = linear_algebra_operations(self.la_matrix.get()) # call LA method
        lines = [ # result formatting
            f"Matrix size: {results['shape'][0]}x{results['shape'][1]}",
            '',
            'Matrix A:',
            format_matrix(results['matrix']),
            '',
            f"Determinant: {results['determinant']:.10f}",
            '',
            'Transpose:',
            format_matrix(results['transpose']),
            '',
        ]

        if results['inverse'] is None: # if matrix doesn't have an inverse, display warning
            lines.append(results['message'])
        else: # if it does, display results
            lines.extend([
                'Inverse:',
                format_matrix(results['inverse']),
                '',
                'Eigenvalues:',
                str(results['eigenvalues']),
                '',
                'Eigenvectors (columns):',
                format_matrix(results['eigenvectors']),
                '',
                results['message'],
            ])

        self.write_output('\n'.join(lines)) # display 

    def _build_system_tab(self): # nonlinear systems --------------------------------------------------------------------------
        frame = ttk.Frame(self.notebook, padding=12) # create tab
        self.notebook.add(frame, text='Nonlinear System')

        self.sys_f1 = tk.StringVar(value='x^2 + y^2 - 4') # tkinter variables
        self.sys_f2 = tk.StringVar(value='x - y - 1')
        self.sys_x0 = tk.StringVar(value='1.5')
        self.sys_y0 = tk.StringVar(value='0.5')
        self.sys_tol = tk.StringVar(value='1e-6')
        self.sys_iter = tk.StringVar(value='25')

        fields = [ # field definitions
            ('f1(x, y):', self.sys_f1),
            ('f2(x, y):', self.sys_f2),
            ('Initial x0:', self.sys_x0),
            ('Initial y0:', self.sys_y0),
            ('Tolerance:', self.sys_tol),
            ('Max iterations:', self.sys_iter),
        ]
        for r, (label, var) in enumerate(fields): # create input boxes
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky='w', padx=5, pady=4)
            ttk.Entry(frame, textvariable=var, width=28).grid(row=r, column=1, sticky='ew', padx=5, pady=4)

        frame.columnconfigure(1, weight=1) # allow stretch
        # button
        ttk.Button(frame, text='Solve System', command=lambda: self._safe_call(self.solve_system)).grid(row=6, column=0, pady=12)

    def solve_system(self): # solve nonlinear systems
        solution, iters, history = newton_system_method(
            self.sys_f1.get(),
            self.sys_f2.get(),
            float(self.sys_x0.get()),
            float(self.sys_y0.get()),
            float(self.sys_tol.get()),
            int(self.sys_iter.get()),
        )
        lines = [ # formatting
            'Newton method for a 2x2 nonlinear system',
            f'x ≈ {solution[0]:.10f}',
            f'y ≈ {solution[1]:.10f}',
            f'Iterations used: {iters}',
            '',
            'Iteration history:',
        ]
        for row in history[:12]: # limit to 12 iterations
            lines.append(str(row))
        if len(history) > 12:
            lines.append('...')
        self.write_output('\n'.join(lines))

    def _build_plot_tab(self): # 2-var plots --------------------------------------------------------------------------
        frame = ttk.Frame(self.notebook, padding=12) # create tab
        self.notebook.add(frame, text='2-Variable Plots')

        self.plot2_expr = tk.StringVar(value='sin(x) * cos(y)') # parameters
        self.plot2_mode = tk.StringVar(value='Contour')
        self.plot2_xmin = tk.StringVar(value='-5')
        self.plot2_xmax = tk.StringVar(value='5')
        self.plot2_ymin = tk.StringVar(value='-5')
        self.plot2_ymax = tk.StringVar(value='5')

        fields = [ # field defs
            ('f(x, y):', self.plot2_expr),
            ('Plot type:', self.plot2_mode),
            ('x min:', self.plot2_xmin),
            ('x max:', self.plot2_xmax),
            ('y min:', self.plot2_ymin),
            ('y max:', self.plot2_ymax),
        ]
        for r, (label, var) in enumerate(fields): # create input boxes
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky='w', padx=5, pady=4)
            if label == 'Plot type:':
                combo = ttk.Combobox(frame, textvariable=var, state='readonly', values=['Contour', 'Surface'])
                combo.grid(row=r, column=1, sticky='ew', padx=5, pady=4)
            else:
                ttk.Entry(frame, textvariable=var, width=28).grid(row=r, column=1, sticky='ew', padx=5, pady=4)

        frame.columnconfigure(1, weight=1) # allow stretch
        # button
        ttk.Button(frame, text='Generate Plot', command=lambda: self._safe_call(self.generate_2d_plot)).grid(row=6, column=0, pady=12)

    def generate_2d_plot(self): # generate the plot
        mode = 'surface' if self.plot2_mode.get() == 'Surface' else 'contour' # check which plot user selected
        self.plot_manager.draw_2d(
            self.plot2_expr.get(),
            float(self.plot2_xmin.get()),
            float(self.plot2_xmax.get()),
            float(self.plot2_ymin.get()),
            float(self.plot2_ymax.get()),
            mode=mode,
        )
        self.write_output(f'Generated a {self.plot2_mode.get().lower()} plot for f(x, y).')


if __name__ == '__main__': # if this file is run, execute
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass
    app = NumericalMethodsGUI(root)
    root.mainloop()
