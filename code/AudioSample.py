from dataclasses import dataclass, field
import os
import torchaudio

class AudioSample:
    """
    Az AudioSample osztály egy audiofájlt reprezentál, amely egy adott könyvtárban található.
    Segítségével betölthető a fájl hossza és mintavételi gyakorisága a kiválasztott könyvtár
    és fájlindex alapján.

    Attribútumok:
        base_path (str): Az alapértelmezett útvonal a fájlok könyvtárához.
        selected_directory (str): A kiválasztott könyvtár neve a `base_path` alatt.
        file_index (int): Az audiofájl indexe a kiválasztott könyvtárban.
        audio_duration (float): A betöltött audiofájl hossza (másodpercben).
        sample_rate (int): A betöltött audiofájl mintavételi gyakorisága (Hz).

    Példa használat:
        audio_sample = AudioSample(base_path="/path/to/audio", selected_directory="samples", file_index=0)
        print(audio_sample.audio_duration)  # Az audio fájl hossza másodpercben
        print(audio_sample.sample_rate)  # Az audio fájl mintavételi gyakorisága
    """

    base_path: str
    selected_directory: str
    file_index: int
    audio_duration: float = field(init=False)
    sample_rate: int = field(init=False)

    def __init__(self, base_path, selected_directory, file_index):
        self.base_path = base_path
        self._selected_directory = selected_directory
        self._file_index = file_index
        self.audio_duration = None
        self.sample_rate = None
        self.n_sample = None
        self.refresh()

    def refresh(self):
        """
        Frissíti az audiofájl hosszát és mintavételi gyakoriságát az aktuális `selected_directory`
        és `file_index` alapján. Automatikusan lefut, amikor a `selected_directory` vagy a `file_index`
        megváltozik.
        """
        self.audio_duration, self.sample_rate, self.n_sample = self._calculate_audio_duration_and_sample_rate()

    @property
    def selected_directory(self):
        """
        Visszaadja a kiválasztott könyvtárat.
        
        Returns:
            str: A kiválasztott könyvtár neve.
        """
        return self._selected_directory

    @selected_directory.setter
    def selected_directory(self, value):
        """
        Beállítja a kiválasztott könyvtárat és frissíti az audiofájl hosszát és mintavételi
        gyakoriságát az új érték alapján.
        
        Args:
            value (str): Az új könyvtár neve.
        """
        self._selected_directory = value
        self.refresh()

    @property
    def file_index(self):
        """
        Visszaadja a fájlindexet.
        
        Returns:
            int: A kiválasztott fájl indexe.
        """
        return self._file_index

    @file_index.setter
    def file_index(self, value):
        """
        Beállítja a fájlindexet és frissíti az audiofájl hosszát és mintavételi
        gyakoriságát az új érték alapján.
        
        Args:
            value (int): Az új fájlindex.
        """
        files = self.get_files()
        if value < 0 or value >= len(files):
            raise IndexError("file_index is out of range for available audio files.")        
        self._file_index = value
        self.refresh()

    def _calculate_audio_duration_and_sample_rate(self):
        """
        Kiszámítja és visszaadja az audiofájl hosszát és mintavételi gyakoriságát.

        Returns:
            tuple: Az audiofájl hossza (float) másodpercben és a mintavételi gyakoriság (int).
        """
        waveform, sample_rate = self.load_waveform()
        audio_duration = waveform.size(0) / sample_rate
        n_sample = waveform.size(0)
        return audio_duration, sample_rate, n_sample

    def load_waveform(self):
        directory_path = os.path.join(self.base_path, self.selected_directory)
        files = self.get_files()  # Uses get_files for modularity
        wav_file_path = os.path.join(directory_path, files[self.file_index])
        waveform, sample_rate = torchaudio.load(wav_file_path)
        return waveform.squeeze(), sample_rate

    def get_files(self):
        """Get all .wav files in the selected directory within the base path."""
        directory_path = os.path.join(self.base_path, self.selected_directory)
        return [f for f in os.listdir(directory_path) if f.endswith('.wav')]    
        
    def get_directories(self):
        """Get all directories within the base path."""
        return [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]        
