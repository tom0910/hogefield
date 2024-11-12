class Spikes:
    """
    Stores and manages the spike data with an observer on threshold updates.
    """

    def __init__(self, threshold):
        self._threshold = threshold  # Use a private variable to store the threshold
        self._threshold_observers = []  # List of observer functions

    @property
    def threshold(self):
        """Get the threshold value."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """Set the threshold value and notify observers of the change."""
        if value != self._threshold:  # Only update if the value changes
            self._threshold = value
            self._notify_observers()

    def add_observer(self, observer_func):
        """Add an observer function that will be called when threshold changes."""
        if observer_func not in self._threshold_observers:
            self._threshold_observers.append(observer_func)

    def _notify_observers(self):
        """Call all observer functions to notify them of the threshold update."""
        for observer in self._threshold_observers:
            observer(self._threshold)

