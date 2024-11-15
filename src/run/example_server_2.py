from flask import Flask, send_file, render_template_string
from io import BytesIO
import numpy as np
from pydub import AudioSegment
import torch
import threading
from tkinter import Tk
from tkinterweb import HtmlFrame  # For displaying HTML content in Tkinter

# Create an instance of Flask web server
app = Flask(__name__)

# Simulated audio_sample class to mimic your audio data
class AudioSample:
    def load_waveform(self):
        # Simulating a sine wave (can be replaced with actual waveform data)
        waveform = torch.sin(torch.linspace(0, 2 * np.pi, 44100))  # 1 second sine wave
        sample_rate = 44100
        return waveform, sample_rate

audio_sample = AudioSample()

@app.route('/')
def index():
    # Load waveform and sample rate from the audio sample
    waveform, sample_rate = audio_sample.load_waveform()

    # Convert waveform to audio file in memory (MP3)
    audio_data = waveform.numpy() * 32767  # Normalize to 16-bit PCM
    audio_segment = AudioSegment(
        audio_data.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
    )
    # Save to in-memory file (BytesIO)
    audio_file = BytesIO()
    audio_segment.export(audio_file, format="mp3")  # Use MP3 format
    audio_file.seek(0)  # Rewind the file pointer

    # Embed the audio player into an HTML page
    return render_template_string("""
        <html>
            <body>
                <h1>Test Audio Display</h1>
                <audio controls>
                    <source src="/audio" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            </body>
        </html>
    """)

@app.route('/audio')
def audio():
    # Generate the audio file dynamically (MP3 format)
    waveform, sample_rate = audio_sample.load_waveform()
    audio_data = waveform.numpy() * 32767
    audio_segment = AudioSegment(
        audio_data.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
    )
    audio_file = BytesIO()
    audio_segment.export(audio_file, format="mp3")  # Use MP3 format
    audio_file.seek(0)  # Rewind the file pointer

    return send_file(audio_file, mimetype="audio/mp3", as_attachment=True, download_name="audio.mp3")

def run_flask_app():
    app.run(host="0.0.0.0", port=5000, use_reloader=False)  # Start Flask server

def display_audio_in_widget_2(waveform, sample_rate, output_frame):
    """
    Display audio playback in a Tkinter frame using an HTML audio widget.

    Args:
        waveform (np.ndarray or torch.Tensor): The audio waveform data.
        sample_rate (int): The sample rate of the audio.
        output_frame (tk.Frame): The Tkinter frame to display the audio player.
    """
    # Start the Flask app in a separate thread so it doesn't block the Tkinter GUI
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    # Create the Tkinter window and HTML frame
    root = Tk()
    root.geometry("800x600")
    frame = HtmlFrame(root, width=800, height=600)
    frame.pack(fill="both", expand=True)

    # Load the URL of the Flask server (localhost)
    frame.load_url("http://localhost:5000")

    # Run the Tkinter main loop
    root.mainloop()

# Test with waveform and sample_rate
waveform, sample_rate = audio_sample.load_waveform()
display_audio_in_widget_2(waveform, sample_rate, None)  # None as output_frame for now
