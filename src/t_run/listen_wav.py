from flask import Flask, render_template_string, request
import os
import webbrowser
from threading import Thread

app = Flask(__name__)

# Folder where audio files are stored
AUDIO_FOLDER = '/project/data/GSC/SpeechCommands/speech_commands_v0.02/backward/'

@app.route('/')
def index():
    # List all audio files in the directory
    audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav')]
    return render_template_string("""
        <html>
            <body>
                <h1>Select an Audio File</h1>
                <form action="/play" method="post">
                    <select name="audio_file">
                        {% for file in audio_files %}
                            <option value="{{ file }}">{{ file }}</option>
                        {% endfor %}
                    </select>
                    <button type="submit">Play</button>
                </form>
            </body>
        </html>
    """, audio_files=audio_files)

@app.route('/play', methods=['POST'])
def play_audio():
    # Get the selected audio file
    audio_file = request.form.get('audio_file')

    # Create the full path to the selected audio file
    file_path = os.path.join(AUDIO_FOLDER, audio_file)

    # Render the audio player HTML
    return render_template_string("""
        <html>
            <body>
                <h1>Now Playing: {{ audio_file }}</h1>
                <audio controls autoplay>
                    <source src="/static/audio/{{ audio_file }}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                <br>
                <a href="/">Back to Selection</a>
            </body>
        </html>
    """, audio_file=audio_file)

# Serve audio files statically
@app.route('/static/audio/<filename>')
def serve_audio(filename):
    file_path = os.path.join(AUDIO_FOLDER, filename)
    return open(file_path, "rb").read(), 200, {'Content-Type': 'audio/wav'}

def run_flask():
    app.run(host="0.0.0.0", port=5001, use_reloader=False)

if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    # Open the app in the default web browser
    webbrowser.open("http://localhost:5001/")
