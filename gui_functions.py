import os
from tkinter import Tk, Toplevel, Button, Scrollbar, END
from tkinter import filedialog, ttk
from datetime import datetime

class CustomFileDialog:
    def __init__(self, root, title, initialdir, filetypes):
        self.selected_files = []
        self.initialdir = initialdir
        # Create a Toplevel window
        self.window = Toplevel(root)
        self.window.title(title)
        self.window.geometry("900x600")
        
        # Treeview for listing files
        self.tree = ttk.Treeview(self.window, columns=("Name", "Date", "Size"), show="headings")  # Updated Treeview import
        self.tree.heading("Name", text="Name", command=lambda: self.sort_files("name"))
        self.tree.heading("Date", text="Date", command=lambda: self.sort_files("date"))
        self.tree.heading("Size", text="Size", command=lambda: self.sort_files("size"))
        self.tree.pack(fill="both", expand=True)
        
        # Scrollbar for Treeview
        scrollbar = Scrollbar(self.tree, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Button to confirm selection
        confirm_button = Button(self.window, text="Select", command=self.select_files)
        confirm_button.pack()

        # Load initial files
        self.load_files(initialdir, filetypes)

    def load_files(self, directory, filetypes):
        self.files = []
        for f in os.listdir(directory):
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path):
                if os.path.splitext(f)[1] in filetypes:
                    mod_time = os.path.getmtime(full_path)
                    formatted_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                    size = round(os.path.getsize(full_path)/1000, 1) #in kB
                    self.files.append((f, formatted_date, size))
                    # mod_time = os.path.getmtime(full_path)
                    # size = os.path.getsize(full_path)
                    # self.files.append((f, mod_time, size))
                    # print(f, os.path.splitext(f))
        
        # Display files initially sorted by name
        self.display_files("name")

    def display_files(self, sort_by):
        self.tree.delete(*self.tree.get_children())
        sorted_files = sorted(self.files, key=lambda x: x[0 if sort_by == "name" else (1 if sort_by == "date" else 2)])
        for f, mod_time, size in sorted_files:
            self.tree.insert("", END, values=(f, mod_time, size))

    def sort_files(self, sort_by):
        # Re-display files sorted by selected criterion
        self.display_files(sort_by)

    def select_files(self):
        selected_items = self.tree.selection()
        self.selected_files = [os.path.join(self.initialdir, self.tree.item(item, "values")[0]) for item in selected_items]
        self.window.destroy()

def custom_filedialog(title, initialdir, filetypes):
    root = Tk()
    root.withdraw()  # Hide the root window
    
    # Launch the custom file dialog
    dialog = CustomFileDialog(root, title, initialdir, filetypes)
    root.wait_window(dialog.window)  # Wait for the dialog to close
    
    # Display selected files
    # print("Selected files:", dialog.selected_files)
    return dialog.selected_files

# if __name__ == "__main__":
#     main()
