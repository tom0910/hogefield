import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Create a Tkinter app
class PlotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tkinter Plots and Widgets")
        self.geometry("900x600")

        # Main Frame
        main_frame = ttk.Frame(self, padding=10, relief="groove")
        main_frame.pack(fill="both", expand=True)

        # Left-side widgets
        self.create_widgets(main_frame)

        # Plot frame (right side)
        plot_frame = ttk.Frame(main_frame, padding=10, relief="sunken")
        plot_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Create the grid of plots
        self.create_plots(plot_frame)

    def create_widgets(self, parent):
        widget_frame = ttk.Frame(parent, padding=10)
        widget_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(widget_frame, text="Controls", font=("Arial", 16)).pack(pady=10)
        ttk.Button(widget_frame, text="Action 1").pack(fill="x", pady=5)
        ttk.Button(widget_frame, text="Action 2").pack(fill="x", pady=5)
        ttk.Button(widget_frame, text="Exit", command=self.quit).pack(fill="x", pady=5)

    def create_plots(self, parent):
        fig = Figure(figsize=(5, 4))
        axes = [
            fig.add_subplot(2, 3, i + 1) if i != 4 else None
            for i in range(6)
        ]

        # Define functions for plots
        x = [i for i in range(10)]
        functions = [
            lambda x: [xi**2 for xi in x],
            lambda x: [xi**3 for xi in x],
            lambda x: [xi + 2 for xi in x],
            lambda x: [2 * xi for xi in x],
            None,
            lambda x: [xi**0.5 for xi in x],
        ]

        # Create plots
        for idx, ax in enumerate(axes):
            if ax:
                y = functions[idx](x)
                ax.plot(x, y)
                ax.set_title(f"Plot {idx + 1}")

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = PlotApp()
    app.mainloop()
