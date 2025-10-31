import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from SkillsExtract import recommend_from_resume

CSV_PATH = r"C:\Users\Lenovo\Desktop\python\MiniProject\internship.csv"

# -------------------- MAIN TKINTER APP --------------------
class ResumeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Internship Recommender")
        self.geometry("1000x500")
        self.resizable(True,True)
        self.resume_path = None

        # Initialize Frames (pages)
        self.frames = {}
        for F in (UploadPage, ResultPage):
            frame = F(self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(UploadPage)

    def show_frame(self, page):
        """Show selected frame."""
        frame = self.frames[page]
        frame.tkraise()

# -------------------- PAGE 1: UPLOAD RESUME --------------------
class UploadPage(tk.Frame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller

        tk.Label(self, text="Upload Your Resume", font=("Helvetica", 18, "bold")).pack(pady=30)

        self.file_label = tk.Label(self, text="No file selected", font=("Arial", 12))
        self.file_label.pack(pady=10)

        tk.Button(self, text="Choose Resume (PDF)", command=self.upload_file, font=("Arial", 12), bg="#0078D7", fg="white").pack(pady=10)
        tk.Button(self, text="Find Internships", command=self.find_internships, font=("Arial", 12), bg="#28a745", fg="white").pack(pady=20)

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Resume",
            filetypes=[("PDF files", "*.pdf")]
        )
        if file_path:
            self.controller.resume_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
        else:
            self.file_label.config(text="No file selected")

    def find_internships(self):
        if not self.controller.resume_path:
            messagebox.showwarning("No File", "Please upload your resume first.")
            return

        self.controller.show_frame(ResultPage)
        self.controller.frames[ResultPage].show_results()

# -------------------- PAGE 2: DISPLAY RESULTS --------------------
class ResultPage(tk.Frame):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller

        tk.Label(self, text="Recommended Internships", font=("Helvetica", 18, "bold")).pack(pady=20)

        frame = tk.Frame(self)
        frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(frame, columns=("Company", "Title", "Location", "Duration", "Stipend"), show='headings', height=15)
        self.tree.heading("Company", text="Company")
        self.tree.heading("Title", text="Internship Title")
        self.tree.heading("Location", text="Location")
        self.tree.heading("Duration", text="Duration")
        self.tree.heading("Stipend", text="Stipend")
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)

        tk.Button(self, text="Back", command=lambda: controller.show_frame(UploadPage), font=("Arial", 12), bg="#dc3545", fg="white").pack(pady=15)

    def show_results(self):
        """Fetch recommendations and show them in table."""
        self.tree.delete(*self.tree.get_children()) 

        resume_path = self.controller.resume_path
        if not os.path.exists(resume_path):
            messagebox.showerror("Error", "Resume file not found.")
            return

        try:
            recommendations = recommend_from_resume(resume_path, CSV_PATH)
            if not recommendations:
                messagebox.showinfo("No Matches", "No internships found matching your resume.")
                return

            for job in recommendations:
                self.tree.insert("", "end", values=(
                    job.get("company_name", ""),
                    job.get("internship_title", ""),
                    job.get("location", ""),
                    job.get("duration", ""),
                    job.get("stipend", "")
                ))
        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong:\n{e}")

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app = ResumeApp()
    app.mainloop()
