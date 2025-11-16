import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import sys


class ScienceMuseumControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("🏛️ Bloomberg Science Museum - Control Panel")

        # Get screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.85)

        self.root.geometry(f"{window_width}x{window_height}")
        self.root.configure(bg='#0f172a')

        # Store active processes
        self.active_processes = {}

        # Create UI
        self.create_header()
        self.create_script_buttons()
        self.create_footer()

    def create_header(self):
        """Create header section"""
        header_frame = tk.Frame(self.root, bg='#1e293b', relief=tk.RAISED, bd=3)
        header_frame.pack(fill=tk.X, padx=15, pady=10)

        # Main title
        title = tk.Label(header_frame,
                         text="🏛️ Bloomberg Science Museum",
                         font=("Arial", 22, "bold"),
                         bg='#1e293b', fg='#00ff88')
        title.pack(pady=5)

        subtitle = tk.Label(header_frame,
                            text="Interactive Computer Vision Learning Center",
                            font=("Arial", 12),
                            bg='#1e293b', fg='#94a3b8')
        subtitle.pack(pady=3)

        # Status
        self.status_label = tk.Label(header_frame,
                                     text="Select a workshop to begin",
                                     font=("Arial", 11, "bold"),
                                     bg='#1e293b', fg='#fbbf24')
        self.status_label.pack(pady=5)

    def create_script_buttons(self):
        """Create buttons for each script"""
        main_frame = tk.Frame(self.root, bg='#0f172a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Define all scripts with their info
        scripts = [
            {
                'id': 'script1',
                'file': 'Script 1 - Black&White Filter On Click.py',
                'title': '⚫⚪ Grayscale Converter',
                'description': 'Learn color to B&W conversion',
                'color': '#8b5cf6',
                'hover': '#7c3aed'
            },
            {
                'id': 'script2',
                'file': 'Script 2 - Color Filter Array.py',
                'title': '🎨 RGB Channel Explorer',
                'description': 'Explore color channels',
                'color': '#ec4899',
                'hover': '#db2777'
            },
            {
                'id': 'script3',
                'file': 'Script 3 - Pixel System.py',
                'title': '🎮 Pixelated Filter Builder',
                'description': 'Create custom filters on 64×64',
                'color': '#3b82f6',
                'hover': '#2563eb'
            },
            {
                'id': 'script4',
                'file': 'Script 4 - In search of a face.py',
                'title': '🔍 Face Detection Stages',
                'description': 'See face detection step-by-step',
                'color': '#14b8a6',
                'hover': '#0d9488'
            },
            {
                'id': 'script5',
                'file': 'Script 5 - I Love Science.py',
                'title': '❤️ I Love Science Mode',
                'description': 'Face detection vs features',
                'color': '#f59e0b',
                'hover': '#d97706'
            },
            {
                'id': 'script6',
                'file': 'Script 6 - Tiger Face Mode.py',
                'title': '🐯 Tiger Face Overlay',
                'description': 'Learn AR face overlays',
                'color': '#ef4444',
                'hover': '#dc2626'
            }
        ]

        # Create 2 columns
        for i, script in enumerate(scripts):
            row = i // 2
            col = i % 2

            self.create_script_card(main_frame, script, row, col)

    def create_script_card(self, parent, script, row, col):
        """Create a card for each script"""
        # Card frame
        card = tk.Frame(parent, bg='#1e293b', relief=tk.RAISED, bd=3)
        card.grid(row=row, column=col, padx=15, pady=15, sticky='nsew')

        # Configure grid weights
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(col, weight=1)

        # Title
        title = tk.Label(card, text=script['title'],
                         font=("Arial", 18, "bold"),
                         bg='#1e293b', fg='white',
                         wraplength=400, justify=tk.LEFT)
        title.pack(pady=(15, 5), padx=15, anchor='w')

        # Description
        desc = tk.Label(card, text=script['description'],
                        font=("Arial", 12),
                        bg='#1e293b', fg='#94a3b8',
                        wraplength=400, justify=tk.LEFT)
        desc.pack(pady=5, padx=15, anchor='w')

        # Button frame
        btn_frame = tk.Frame(card, bg='#1e293b')
        btn_frame.pack(pady=15, padx=15, fill=tk.X)

        # Launch button
        launch_btn = tk.Button(btn_frame,
                               text="▶️ Launch",
                               command=lambda: self.launch_script(script),
                               font=("Arial", 13, "bold"),
                               bg=script['color'],
                               fg='white',
                               width=12,
                               height=2,
                               relief=tk.RAISED,
                               bd=3,
                               cursor='hand2',
                               activebackground=script['hover'])
        launch_btn.pack(side=tk.LEFT, padx=5)

        # Stop button
        stop_btn = tk.Button(btn_frame,
                             text="⏹️ Stop",
                             command=lambda: self.stop_script(script),
                             font=("Arial", 13, "bold"),
                             bg='#64748b',
                             fg='white',
                             width=12,
                             height=2,
                             relief=tk.RAISED,
                             bd=3,
                             cursor='hand2',
                             activebackground='#475569',
                             state=tk.DISABLED)
        stop_btn.pack(side=tk.LEFT, padx=5)

        # Status indicator
        status = tk.Label(btn_frame,
                          text="●",
                          font=("Arial", 20),
                          bg='#1e293b',
                          fg='#64748b')
        status.pack(side=tk.LEFT, padx=10)

        # Store button references
        script['launch_btn'] = launch_btn
        script['stop_btn'] = stop_btn
        script['status'] = status

    def launch_script(self, script):
        """Launch a script"""
        script_id = script['id']
        script_file = script['file']

        # Check if already running
        if script_id in self.active_processes:
            messagebox.showwarning("Already Running",
                                   f"{script['title']} is already running!")
            return

        # Check if file exists
        if not os.path.exists(script_file):
            messagebox.showerror("File Not Found",
                                 f"Cannot find: {script_file}\n\nMake sure the script is in the same directory as this control panel.")
            return

        try:
            # Launch script as subprocess
            process = subprocess.Popen([sys.executable, script_file],
                                       creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)

            # Store process
            self.active_processes[script_id] = process

            # Update UI
            script['launch_btn'].config(state=tk.DISABLED)
            script['stop_btn'].config(state=tk.NORMAL)
            script['status'].config(fg='#00ff88')  # Green

            self.status_label.config(text=f"✅ {script['title']} launched successfully!")

        except Exception as e:
            messagebox.showerror("Launch Error",
                                 f"Failed to launch {script['title']}:\n{str(e)}")

    def stop_script(self, script):
        """Stop a running script"""
        script_id = script['id']

        if script_id not in self.active_processes:
            return

        try:
            # Terminate process
            process = self.active_processes[script_id]
            process.terminate()

            # Wait for termination
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()

            # Remove from active processes
            del self.active_processes[script_id]

            # Update UI
            script['launch_btn'].config(state=tk.NORMAL)
            script['stop_btn'].config(state=tk.DISABLED)
            script['status'].config(fg='#64748b')  # Gray

            self.status_label.config(text=f"⏹️ {script['title']} stopped")

        except Exception as e:
            messagebox.showerror("Stop Error",
                                 f"Failed to stop {script['title']}:\n{str(e)}")

    def create_footer(self):
        """Create footer with controls"""
        footer_frame = tk.Frame(self.root, bg='#1e293b', relief=tk.RAISED, bd=3)
        footer_frame.pack(fill=tk.X, padx=20, pady=20)

        # Stop all button
        stop_all_btn = tk.Button(footer_frame,
                                 text="⏹️ Stop All Workshops",
                                 command=self.stop_all,
                                 font=("Arial", 14, "bold"),
                                 bg='#dc2626',
                                 fg='white',
                                 width=25,
                                 height=2,
                                 relief=tk.RAISED,
                                 bd=4,
                                 cursor='hand2',
                                 activebackground='#b91c1c')
        stop_all_btn.pack(side=tk.LEFT, padx=20, pady=15)

        # Info
        info = tk.Label(footer_frame,
                        text="💡 Tip: Each workshop opens in a new window. Close this panel to exit all workshops.",
                        font=("Arial", 11),
                        bg='#1e293b',
                        fg='#94a3b8')
        info.pack(side=tk.LEFT, padx=20)

    def stop_all(self):
        """Stop all running scripts"""
        if not self.active_processes:
            messagebox.showinfo("No Active Workshops", "No workshops are currently running.")
            return

        count = len(self.active_processes)

        # Stop all processes
        for script_id in list(self.active_processes.keys()):
            process = self.active_processes[script_id]
            try:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            except:
                pass

        self.active_processes.clear()

        # Update all buttons
        for widget in self.root.winfo_children():
            self.reset_all_buttons(widget)

        self.status_label.config(text=f"⏹️ Stopped {count} workshop(s)")

    def reset_all_buttons(self, widget):
        """Recursively reset all buttons"""
        for child in widget.winfo_children():
            self.reset_all_buttons(child)

    def on_closing(self):
        """Handle window closing"""
        if self.active_processes:
            if messagebox.askokcancel("Quit",
                                      f"{len(self.active_processes)} workshop(s) are still running.\nStop all and exit?"):
                self.stop_all()
                self.root.destroy()
        else:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ScienceMuseumControlPanel(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()