#tk_widgets_setup
import tkinter as tk
from tkinter import ttk
import config.config as config
import utils.functional as FU


class ChoiceWidget:
    def __init__(self, parent, variable, label_text="Select:", options=None):
        if options is None:
            options = []  # Default to an empty list if no options are provided
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("`options` must be a list with at least two choices.")

        self.var = variable
        self.label = ttk.Label(parent, text=label_text)
        self.radio_buttons = [
            ttk.Radiobutton(parent, text=label, value=value, variable=self.var)
            for label, value in options
        ]

    def grid(self, row, column, **kwargs):
        self.label.grid(row=row, column=column, **kwargs)
        for i, radio in enumerate(self.radio_buttons):
            radio.grid(row=row, column=column + 1 + i, **kwargs)

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)

    def bind(self, callback):
        if callback is not None:
            self.var.trace_add("write", lambda *args: callback())
        else:
            raise ValueError("Callback function cannot be None")
    # usage:
    # Accepts any number of options as a list of tuples (label, value)
    # mode_var = tk.StringVar(value="Mode A")
    # mode_widget = ChoiceWidget(
    #     parent=widget_frame,
    #     variable=mode_var,
    #     label_text="Select Mode:",
    #     options=[("Mode A", "A"), ("Mode B", "B"), ("Mode C", "C")]
    # )
    # mode_widget.grid(row=1, column=0, padx=5, pady=5)


class PowerToggle:
    def __init__(self, parent, variable, label_text="select=>"):
        self.var = variable
        self.label = ttk.Label(parent, text=label_text)
        self.radio_power = ttk.Radiobutton(parent, text="Power", value=2.0, variable=self.var)
        self.radio_magnitude = ttk.Radiobutton(parent, text="Magnitude", value=1.0, variable=self.var)

    def grid(self, row, column, **kwargs):
        self.label.grid(row=row, column=column, **kwargs)
        self.radio_power.grid(row=row, column=column + 1, **kwargs)
        self.radio_magnitude.grid(row=row, column=column + 2, **kwargs)

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(float(value))

    def bind(self, callback):
        if callback is not None:
            self.var.trace_add("write", lambda *args: callback())
        else:
            raise ValueError("Callback function cannot be None")   
        
import tkinter as tk
from tkinter import ttk


# class EntryWidget:
#     def __init__(self, parent, variable, label_text="Enter:", default_value=10):
#         self.var = variable
#         # Dynamically set the default value
#         if isinstance(self.var, tk.StringVar):
#             self.var.set(str(default_value))
#         elif isinstance(self.var, (tk.IntVar, tk.DoubleVar)):
#             self.var.set(default_value)
#         else:
#             raise TypeError("Unsupported variable type: Use StringVar, IntVar, or DoubleVar.")
        
#         print(f"Default value set: {self.var.get()}")
#         self.label = ttk.Label(parent, text=label_text)
#         self.entry = ttk.Entry(parent, textvariable=self.var, width=10)
#         self.entry.insert(0, "write me babe")

#     def grid(self, row, column, **kwargs):
#         self.label.grid(row=row, column=column, **kwargs)
#         self.entry.grid(row=row+1, column=column, **kwargs)

#     def get(self):
#         return self.var.get()

#     def set(self, value):
#         self.var.set(value)

#     def bind(self, callback):
#         if callback is not None:
#             self.var.trace_add("write", lambda *args: callback(self.get()))
#         else:
#             raise ValueError("Callback function cannot be None")

  

# Function to set up widgets using Tkinter
def create_widgets(audio_sample, mel_config, spikes_data, widget_frame):

    # Directory Dropdown (Combobox in Tkinter)
    directory_dropdown_label = ttk.Label(widget_frame, text='Directory:')
    directory_dropdown = ttk.Combobox(widget_frame, values=audio_sample.get_directories())
    directory_dropdown.set(config.DEFAULT_DIRECTORY)

    # File Index Slider
    file_index_var = tk.IntVar(value=10)
    file_slider_label = ttk.Label(widget_frame, text='File Index:')
    file_slider = tk.Scale(widget_frame, from_=0, to=len(audio_sample.get_files()) - 1, orient='horizontal',variable=file_index_var)
    file_slider_entry = ttk.Entry(widget_frame, textvariable=file_index_var, width=5)

    n_fft_var = tk.IntVar(value=config.DEFAULT_N_FFT)
    n_fft_label = ttk.Label(widget_frame, text='n_fft:')
    # n_fft_slider = tk.Scale(widget_frame, from_=1, to=1024, orient='horizontal', variable=n_fft_var)
    n_fft_slider = tk.Scale(widget_frame, from_=256, to=2048, orient='horizontal', variable=n_fft_var, resolution=256)
    n_fft_entry = ttk.Entry(widget_frame, textvariable=n_fft_var, width=5)

    
    hop_length_var = tk.IntVar(value=config.DEFAULT_HOP_LENGTH)
    hop_length_label = ttk.Label(widget_frame, text='Hop Length:')
    hop_length_slider = tk.Scale(widget_frame, from_=1, to=600, orient='horizontal', variable=hop_length_var)
    hop_length_entry = ttk.Entry(widget_frame, textvariable=hop_length_var, width=5)

    # Number of Mel Bands Slider    
    n_mels_var = tk.IntVar(value=config.DEFAULT_N_MELS)
    n_mels_label = ttk.Label(widget_frame, text='n_mels:')
    n_mels_slider = tk.Scale(widget_frame, from_=1, to=88, orient='horizontal', variable=n_mels_var)
    n_mels_entry = ttk.Entry(widget_frame, textvariable=n_mels_var, width=5)    


    # f_min Slider
    f_min_label = ttk.Label(widget_frame, text='f_min:')
    f_min_slider = tk.Scale(widget_frame, from_=0, to=8000, resolution=100, orient='horizontal')
    f_min_slider.set(config.DEFAULT_F_MIN)

    # f_max Slider
    f_max_label = ttk.Label(widget_frame, text='f_max:')
    f_max_slider = tk.Scale(widget_frame, from_=8000, to=16000, resolution=100, orient='horizontal')
    f_max_slider.set(config.DEFAULT_F_MAX)

    # Create PowerToggle with DoubleVar
    power_label = ttk.Label(widget_frame, text='Amplitude in what?:')
    power_var = tk.DoubleVar()  # Use DoubleVar to ensure float handling
    power_var.set(2.0 if config.DEFAULT_POWER == 2.0 else 1.0)  # Set the default value
    power_toggle = PowerToggle(widget_frame, power_var)    
        
    filter_choice_var = tk.StringVar(value=config.DEFAULT_FILTER_CHOICE)
    filter_choice_widget = ChoiceWidget(
        parent=widget_frame,
        variable=filter_choice_var,
        label_text="Select Mode:",
        options=[("standard", config.FILTER_VALUE1), ("custom", config.FILTER_VALUE2), ("narrowband", config.FILTER_VALUE3)]
    )

    
    mel_filter_plot_var = tk.StringVar(value=config.DEFAULT_FILTER_SPCTRGRM_PLT_CHOICE)
    mel_filter_plot_radio_widget = ChoiceWidget(
        parent=widget_frame,
        variable=mel_filter_plot_var,
        label_text="Select Plot Type:",
        options=[("spectrogram", config.DEFAULT_SPCTRGRM_PLT), ("filter", config.DEFAULT_FILTER_PLT)]
    )
    
    hop_length_var = tk.IntVar(value=config.DEFAULT_HOP_LENGTH)
    hop_length_label = ttk.Label(widget_frame, text='Hop Length:')
    hop_length_slider = tk.Scale(widget_frame, from_=1, to=512, orient='horizontal', variable=hop_length_var)
    hop_length_entry = ttk.Entry(widget_frame, textvariable=hop_length_var, width=5)    

    threshold_var = tk.DoubleVar(value=config.DEFAULT_THRESHOLD)
    threshold_label = ttk.Label(widget_frame, text='Threshold:')
    threshold_slider = tk.Scale(widget_frame, from_=config.DEFAULT_THRESHOLD_MIN, to=config.DEFAULT_THRESHOLD_MAX, resolution=config.DEFAULT_THRESHOLD_STEP, orient='horizontal', variable=threshold_var, length=175)
    threshold_entry = ttk.Entry(widget_frame, textvariable=threshold_var, width=10)
    
    print("After initialization:", type(threshold_entry))  # Expect <class 'tkinter.ttk.Entry'>

    spike_plot_radio_var = tk.StringVar(value=config.DEFAULT_SPIKE_PLT_PICK)
    spike_plot_radio_widget = ChoiceWidget(
        parent=widget_frame,
        variable=spike_plot_radio_var,
        label_text="Choose plot:",
        options=[("spike trains plot", config.DEFAULT_SPIKE_PLT_PICK), ("distribution plot", config.DEFAULT_DIST_PLT_PICK)]
    )

    # Spike Duration Slider
    spike_period_slider_label = ttk.Label(widget_frame, text='range:')
    spike_periode_slider = tk.Scale(widget_frame, from_=0, to=mel_config.step_duration, resolution=10, orient='horizontal')
    spike_periode_slider.set(150)

    # Channel Slider
    channel_slider_label = ttk.Label(widget_frame, text='Ch:')
    channel_slider = tk.Scale(widget_frame, from_=1, to=mel_config.n_mels, resolution=1, orient='horizontal')
    channel_slider.set(1)

    # Frequency Calculation and Label
    def freq_calc():
        _, _, spikes, _, _ = FU.generate_spikes(audio_sample, mel_config, spikes_data.threshold)
        max_start, max_stop, max_count = FU.find_max_interval(spikes, channel_slider.get(), spike_periode_slider.get())
        mel_time_resolution = mel_config.time_resolution
        range_steps = max_stop - max_start
        occurance_freq = max_count / (mel_time_resolution * range_steps)
        return occurance_freq, mel_time_resolution, max_start, max_stop, int(max_count)

    spk_freq_label = ttk.Label(widget_frame, text="")

    def update_label():
        occurance_freq, mel_time_resolution, max_start, max_stop, max_count = freq_calc()
        spk_freq_label.config(
            text=f"ch:{channel_slider.get()} rate:{occurance_freq:.1f}[Hz]\n"
                 f"{max_start} _ {max_stop} cnt:{max_count}\n"
                 f"dt:{mel_time_resolution * 1000:.2f}[msec] 1/dt:{1 / (mel_time_resolution * 1000):.1f}kHz"
        )
    # def update_treshold():
    #     spikes_data.treshold = threshold_var
    #     update_label()
    
    spike_periode_slider.config(command=lambda _: update_label())
    channel_slider.config(command=lambda _: update_label())
    hop_length_slider.config(command=lambda _: update_label())
    threshold_slider.config(command=lambda _: update_label())
        
    save_button_widget = tk.Button(widget_frame, text="Save Parms.")
    save_matdata_widget = tk.Button(widget_frame, text="Save .mat")
    
    # Add an Entry widget for the filename
    filename_label = tk.Label(widget_frame, text="ID:")
    filename_entry = tk.Entry(widget_frame, width=10)

    # Bind the save button to the function, passing the Entry widget as an argument
    #save_button.bind("<Button-1>", lambda event: on_save_button_click(filename_entry))    
            
    # Arrange widgets in a grid layout for better UI
    widgets = [
        (directory_dropdown_label, directory_dropdown,None),
        (file_slider_label, file_slider, file_slider_entry),
        # (n_fft_label, n_fft_input, None),
        (n_fft_label, n_fft_slider, n_fft_entry),
        (hop_length_label, hop_length_slider,  hop_length_entry),
        (n_mels_label, n_mels_slider, n_mels_entry),
        (f_min_label, f_min_slider, None),
        (f_max_label, f_max_slider, None),
        (power_label, power_toggle, None),
        (threshold_label, threshold_slider, threshold_entry),
        (spike_plot_radio_widget, None, None),
        (spike_period_slider_label, spike_periode_slider, None),
        (channel_slider_label, channel_slider, None),
        (spk_freq_label, None,None),
        (filter_choice_widget,None,None),
        (mel_filter_plot_radio_widget,None,None),
        (save_button_widget,filename_label,filename_entry),
        (save_matdata_widget,None,None)
    ]
        

    row = 0
    for label, widget, entry in widgets:
        if label:
            label.grid(row=row, column=0, padx=5, pady=5, sticky='w')
        if widget:
            widget.grid(row=row, column=1, padx=5, pady=5, sticky='w')
        if entry:
            entry.grid(row=row, column=2, padx=5, pady=5, sticky='w')
        row += 1

    return (
        directory_dropdown, 
        file_slider, 
        file_slider_entry,
        # n_fft_input, 
        n_fft_label, n_fft_slider, n_fft_entry,
        hop_length_slider, 
        n_mels_slider, 
        f_min_slider, 
        f_max_slider, 
        power_toggle, 
        threshold_slider, 
        spike_plot_radio_widget, 
        threshold_entry,
        spike_periode_slider,spk_freq_label,channel_slider, 
        filter_choice_widget,
        mel_filter_plot_radio_widget, 
        spk_freq_label,
        hop_length_entry, n_mels_entry,
        save_button_widget,
        filename_entry,
        save_matdata_widget
    )
    
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def initialize_plots(frames, figsize=(4, 4)):
    """
    Initializes figures, axes, and canvases for multiple plots.
    
    Args:
        frames (list): List of Tkinter frames where plots will be embedded.
        figsize (tuple): Size of the Matplotlib figures.

    Returns:
        tuple: A list of figures, axes, and canvases.
    """
    figs, axes, canvases = [], [], []
    
    for frame in frames:
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Initialize _colorbar attribute for each axis
        ax._colorbar = None
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        figs.append(fig)
        axes.append(ax)
        canvases.append(canvas)
    
    return figs, axes, canvases

# Create the plots frame without embedding the waveform plot
def create_plots_frame(root):
    
    # Create a frame for widgets (left side)
    widget_frame = ttk.Frame(root, padding=10, relief="ridge")
    widget_frame.grid(row=0, column=0, sticky="ns")  # Left-side frame
        
    # Create a frame for the plots
    plot_frame = ttk.Frame(root, padding=10, relief="sunken")
    plot_frame.grid(row=0, column=1, sticky="nsew")

    # Define 2x3 grid
    subframes = [ttk.Frame(plot_frame) for _ in range(6)]
    for idx, frame in enumerate(subframes):
        row, col = divmod(idx, 3)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        
    # Set grid weights for dynamic resizing
    for i in range(2):  # Two rows
        plot_frame.rowconfigure(i, weight=1)
    for i in range(3):  # Three columns
        plot_frame.columnconfigure(i, weight=1)        
    
    out_audio_wave_plt = subframes[0]  
    out_mel_sptgrm_plt = subframes[1]  
    out_spike_raster_plt = subframes[2]
    out_rev_spike_raster_plt = subframes[3]
    out_rev_mel_sptgrm_plt = subframes[4]  
    out_rev_play = subframes[5] 
    return widget_frame, out_audio_wave_plt, out_mel_sptgrm_plt, out_spike_raster_plt, out_rev_spike_raster_plt, out_rev_mel_sptgrm_plt, out_rev_play  

def create_plots_frame_simple(root):
    
    # Configure the grid for `root`
    root.columnconfigure(0, weight=0)  # Fixed width for `widget_frame`
    root.columnconfigure(1, weight=1)  # Expand `plot_frame` to fill remaining space
    root.rowconfigure(0, weight=1)     # Allow vertical resizing
    
    # Create a frame for widgets (left side)
    widget_frame = ttk.Frame(root, padding=10, relief="ridge")
    widget_frame.grid(row=0, column=0, sticky="ns")  # Left-side frame

    # Create a frame for the plots
    plot_frame = ttk.Frame(root, padding=10, relief="sunken")
    plot_frame.grid(row=0, column=1, sticky="nsew")

    # Create subframes
    subframes = [ttk.Frame(plot_frame) for _ in range(7)]  # 4 for 2x2 and 3 for 1x3
 

    # Arrange 2x2 grid (top part), each spanning two columns
    subframes[0].grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")  # Top-left
    subframes[1].grid(row=0, column=3, columnspan=3, padx=5, pady=5, sticky="nsew")  # Top-right
    subframes[3].grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")  # Bottom-left of 2x2
    subframes[2].grid(row=1, column=3, columnspan=3, padx=5, pady=5, sticky="nsew")  # Bottom-right of 2x2

    # Arrange 1x3 grid (bottom part)
    # Arrange 1x3 grid (bottom part)
    subframes[4].grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")  # Bottom-left
    subframes[5].grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="nsew")  # Bottom-center
    subframes[6].grid(row=2, column=4, columnspan=2, padx=5, pady=5, sticky="nsew")  # Bottom-right (spans 2 columns)


    # Configure row and column weights for dynamic resizing
    for col in range(6):  # 4 columns for 2x2 and 1x3 grids
        plot_frame.columnconfigure(col, weight=1)

    plot_frame.rowconfigure(0, weight=1)  # First row (2x2 grid)
    plot_frame.rowconfigure(1, weight=1)  # Second row (2x2 grid)
    plot_frame.rowconfigure(2, weight=1)  # Third row (1x3 grid)

    out_audio_wave_plt = subframes[0]
    out_mel_sptgrm_plt = subframes[1]
    out_spike_raster_plt = subframes[4]
    out_rev_spike_raster_plt = subframes[5]
    out_rev_mel_sptgrm_plt = subframes[6]
    out_rev_play = subframes[2]
    out_rev_play2 = subframes[3]

    return widget_frame, out_audio_wave_plt, out_mel_sptgrm_plt, out_spike_raster_plt, out_rev_spike_raster_plt, out_rev_mel_sptgrm_plt, out_rev_play,out_rev_play2


def create_plots_frame_simple2(root):
    
    # Configure the grid for `root`
    root.columnconfigure(0, weight=0)  # Fixed width for `widget_frame`
    root.columnconfigure(1, weight=1)  # Expand `plot_frame` to fill remaining space
    root.rowconfigure(0, weight=1)     # Allow vertical resizing
    
    # Create a frame for widgets (left side)
    widget_frame = ttk.Frame(root, padding=10, relief="ridge")
    widget_frame.grid(row=0, column=0, sticky="ns")  # Left-side frame

    # Create a frame for the plots
    plot_frame = ttk.Frame(root, padding=10, relief="sunken")
    plot_frame.grid(row=0, column=1, sticky="nsew")

    # Define a new grid layout
    subframes = [ttk.Frame(plot_frame) for _ in range(6)]

    # Set layout for `out_audio_wave_plt` (double-wide)
    subframes[0].grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

    # Set layout for the other frames
    subframes[1].grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

    # Set grid weights for dynamic resizing
    plot_frame.rowconfigure(0, weight=1)  # First row
    plot_frame.rowconfigure(1, weight=1)  # Second row
    
    for i in range(3):  # Three columns
        plot_frame.columnconfigure(i, weight=1)

    out_audio_wave_plt = subframes[0]
    out_spike_raster_plt = subframes[1]


    return widget_frame, out_audio_wave_plt, out_spike_raster_plt

