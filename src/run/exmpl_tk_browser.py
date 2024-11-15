import tkinter as tk
from tkinterweb import HtmlFrame

def open_browser_in_tkinter():
    # Create the Tkinter root window
    root = tk.Tk()
    root.title("Simple Browser Inside Tkinter")
    
    # Create a frame to hold the web browser content
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)
    
    # Create the HtmlFrame widget that will act as the browser
    browser = HtmlFrame(frame, horizontal_scrollbar="auto")
    browser.pack(fill="both", expand=True)
    
    # Open a webpage (replace with the URL you want)
    url = "https://www.example.com"  # You can replace this with any URL
    browser.load_url(url)
    
    # Start the Tkinter event loop
    root.mainloop()

# Run the function to open the browser within Tkinter
open_browser_in_tkinter()
