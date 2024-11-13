import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Set the interactive backend for matplotlib
plt.switch_backend('TkAgg')

# Sample data to plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Initialize widgets
frequency_slider = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Frequency:')
amplitude_slider = widgets.FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description='Amplitude:')
plot_button = widgets.Button(description="Update Plot")

# Function to update the plot
def update_plot(change):
    frequency = frequency_slider.value
    amplitude = amplitude_slider.value
    plt.figure(figsize=(10, 5))
    plt.plot(x, amplitude * np.sin(frequency * x))
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Interactive Sine Wave Plot')
    plt.grid(True)
    plt.show()

# Attach the function to the button
plot_button.on_click(update_plot)

# Display widgets and initial plot
display(frequency_slider, amplitude_slider, plot_button)
update_plot(None)

