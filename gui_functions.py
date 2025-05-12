from tkinter import ttk, Tk, Toplevel, Button, Scrollbar, Label, Frame, Canvas, END
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from wsxm_read import get_wsxm_filelist, wsxm_readgeneral

class CustomFileDialog:
    def __init__(self, root, title, initialdir, file_ext, channels, flatten_chan, img_dirs):
        self.selected_file_dict = {}
        self.initialdir = Path(initialdir)
        self.channels = channels
        self.flatten_chan = flatten_chan
        self.img_dirs = img_dirs

        # Create a Toplevel window
        self.window = Toplevel(root)
        self.window.title(title)
        self.window.geometry("1200x600")

        # Create a main layout frame
        main_frame = Frame(self.window)
        main_frame.pack(fill="both", expand=True)

        # File Treeview (Left Panel)
        self.tree = ttk.Treeview(main_frame, columns=("Name", "Date", "Size"), show="headings")
        # Define column widths
        self.tree.column("Name", width=450, anchor="w")  # Set a larger width for Name
        self.tree.column("Date", width=100, anchor="center")  # Adjust Date column width
        self.tree.column("Size", width=20, anchor="e")  # Adjust Size column
        self.tree.heading("Name", text="Name", command=lambda: self.sort_files("name"))
        self.tree.heading("Date", text="Date", command=lambda: self.sort_files("time"))
        self.tree.heading("Size", text="Size", command=lambda: self.sort_files("size"))
        self.tree.pack(side="left", fill="both", expand=True)

        # Scrollbar for Treeview
        scrollbar = Scrollbar(self.tree, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar_x = Scrollbar(self.tree, orient="horizontal", command=self.tree.xview)
        scrollbar_x.pack(side="bottom", fill="x")
        self.tree.configure(xscrollcommand=scrollbar_x.set)

        # Right-side Preview Panel (Scrollable)
        preview_frame = Frame(main_frame, width=450)
        preview_frame.pack(side="right", fill="y")  # Remove "expand=True" to prevent stretching

        # Create a frame inside preview_frame to hold the canvas and scrollbars
        container = Frame(preview_frame, width=450)
        container.pack(fill="both", expand=True)
        
        # Create a Canvas for scrolling
        self.canvas = Canvas(container, width=450)
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Create a scrollable frame inside the canvas
        self.scroll_frame = Frame(self.canvas, width=450)
        self.window_id = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        
        # Vertical scrollbar
        self.scrollbar_y = Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollbar_y.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set)
        
        # Horizontal scrollbar
        self.scrollbar_x = Scrollbar(preview_frame, orient="horizontal", command=self.canvas.xview)
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set)
                    
        def on_mouse_wheel(event):
            # print(event.delta)
            self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
        
        self.canvas.bind_all("<MouseWheel>", on_mouse_wheel)  # Enable mouse scroll
        self.canvas.bind("<Button-5>", on_mouse_wheel) #only triggers even if cursor not on any figure
        
        self.tooltip = Label(self.window, bg="lightyellow", relief="solid", borderwidth=1)

        # Button to confirm selection
        confirm_button = Button(self.window, text="Select", command=self.select_files)
        confirm_button.pack()

        # List of preview image labels
        self.preview_images = []  # To store references
        self.image_labels = []  # To hold Label widgets

        # Load initial files
        self.file_df = get_wsxm_filelist(self.initialdir, file_ext = file_ext)
        # Display files initially sorted by time
        self.display_files("time")

        # Bind file selection event
        self.tree.bind("<<TreeviewSelect>>", self.on_file_select)

    def display_files(self, sort_by):
        self.tree.delete(*self.tree.get_children())
        self.file_df.sort_values(by=[sort_by], inplace=True, ignore_index=True)
        for row in self.file_df.itertuples():
            self.tree.insert("", END, values=(row.name, row.time, f"{row.size} kB   "))

    def sort_files(self, sort_by):
        self.display_files(sort_by)

    def select_files(self):
        selected_items = self.tree.selection()
        self.selected_file_dict = {}
        for item in selected_items:
            file_df_filt = self.file_df.query(f"name == '{self.tree.item(item, 'values')[0]}'").iloc[0] 
            # print(file_df_filt)
            self.selected_file_dict[file_df_filt['name']] = file_df_filt.path
        self.window.destroy()

    def on_file_select(self, event):
        """Update preview images when file selection changes."""
        selected_items = self.tree.selection()
        selected_files = [self.file_df.query(f"name == '{self.tree.item(item, 'values')[0]}'").path.iloc[0] for item in selected_items]
        self.update_image_previews(selected_files)

    def update_image_previews(self, file_paths):
        """Efficiently display Plotly figures in Tkinter as static images."""
        for label in self.image_labels:
            label.destroy()
        self.preview_images.clear()
        self.image_labels.clear()
    
        fixed_size = (150, 150)  # Adjust size for consistency
    
        for index, file_path in enumerate(file_paths):
            try:
                fig_dict = wsxm_readgeneral(file_path, channels=self.channels, flatten_chan=self.flatten_chan, 
                                           img_dirs=self.img_dirs)#['Topography', 'Amplitude'])
                col = 0
                for key, fig in sorted(fig_dict.items()):
                    # Embed Matplotlib figure in Tkinter
                    canvas = FigureCanvasTkAgg(fig, master=self.scroll_frame)
                    widget = canvas.get_tk_widget()
                    widget.grid(row=index, column=col, padx=5, pady=5)
                    widget.bind("<Motion>", lambda e, k=key: self.show_tooltip(e, k))
                    widget.bind("<Leave>", lambda e: self.hide_tooltip())
                    self.image_labels.append(widget)
                    col += 1
    
            except Exception as e:
                print(f"Error generating preview for {file_path}: {e}")
    
        self.scroll_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def show_tooltip(self, event, text):
        """Show tooltip near the hovered preview."""
        self.tooltip.config(text=text)
        self.tooltip.place(x=event.x_root - self.window.winfo_rootx() + 10, 
                            y=event.y_root - self.window.winfo_rooty() + 10)
        self.tooltip.lift()  # Bring tooltip to front

    def hide_tooltip(self):
        self.tooltip.place_forget()


def custom_filedialog(title='', initialdir='/', file_ext=[], channels=[], flatten_chan=[], img_dirs=[]):
    root = Tk()
    root.withdraw()  # Hide the root window
    
    # Launch the custom file dialog
    dialog = CustomFileDialog(root, title, initialdir, file_ext, channels, flatten_chan, img_dirs)
    root.wait_window(dialog.window)  # Wait for the dialog to close
    root.destroy()
    
    return dialog.selected_file_dict