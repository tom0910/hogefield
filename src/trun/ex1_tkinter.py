import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Create the main Tkinter window
root = tk.Tk()
root.title("Tkinter Widgets UI Example")
root.geometry("800x600")

# Frame for Control Widgets
control_frame = ttk.Frame(root)
control_frame.grid(row=0, column=0, sticky='nw', padx=10, pady=10)

# Frame for Figures
figure_frame = ttk.Frame(root)
figure_frame.grid(row=0, column=1, sticky='ne', padx=10, pady=10)

# Add Widgets to Control Frame
# Dropdown (OptionMenu in Tkinter)
selected_option = tk.StringVar()
dropdown = ttk.Combobox(control_frame, textvariable=selected_option)
dropdown['values'] = ("Option 1", "Option 2", "Option 3")
dropdown.set("Option 1")
dropdown.grid(row=0, column=0, pady=5)

# Slider (Scale in Tkinter)
slider = tk.Scale(control_frame, from_=0, to=100, orient='horizontal', label='Slider')
slider.grid(row=1, column=0, pady=5)

# Button to Update Plot
def update_plot():
    frequency = slider.get()
    y = np.sin(frequency * x)
    ax.clear()
    ax.plot(x, y)
    canvas.draw()

update_button = ttk.Button(control_frame, text="Update Plot", command=update_plot)
update_button.grid(row=2, column=0, pady=5)

# Create a Figure and Plot in the Figure Frame
x = np.linspace(0, 10, 100)
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
ax.plot(x, np.sin(x))

# Embed the figure to the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=figure_frame)
canvas.draw()
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0)

# Arrange Layout with Frame and Widgets
control_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nw')
figure_frame.grid(row=0, column=1, padx=10, pady=10, sticky='ne')

# Run the main Tkinter event loop
root.mainloop()

