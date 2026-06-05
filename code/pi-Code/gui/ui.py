import customtkinter as ctk
from customtkinter import filedialog
from tkinter import messagebox
import numpy as np
import librosa
import tensorflow as tf
import os

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

MODEL_PATH = "../../model/urban_audio_edge.tflite"
SR = 16000
DURATION = 3
N_MELS = 64

LABELS = [
    "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", 
    "Drilling", "Engine Idling", "Gun Shot", "Siren", "Street Music", "Jackhammer"
]

LABEL_ICONS = {
    "Air Conditioner": "❄️", "Car Horn": "📢", "Children Playing": "🧒",
    "Dog Bark": "🐕", "Drilling": "🔨", "Engine Idling": "🚗",
    "Gun Shot": "🔫", "Siren": "🚨", "Street Music": "🎵", "Jackhammer": "⛏️"
}

# --- LOAD MODEL ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(file_path):
    audio_data, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    if len(audio_data) < SR * DURATION:
        audio_data = np.pad(audio_data, (0, SR * DURATION - len(audio_data)))
    mel = librosa.feature.melspectrogram(y=audio_data, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)

class ModernAudioClassifier(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Urban Sound Classifier")
        self.geometry("900x580")
        self.minsize(800, 500)
        
        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (450)
        y = (self.winfo_screenheight() // 2) - (275)
        self.geometry(f'+{x}+{y}')
        
        self.selected_file = None
        self.is_processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Main container
        self.main_container = ctk.CTkFrame(self, fg_color=("gray95", "gray17"), corner_radius=25)
        self.main_container.grid(row=0, column=0, padx=25, pady=25, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.header.grid(row=0, column=0, padx=30, pady=(25, 10), sticky="ew")
        
        ctk.CTkLabel(
            self.header, 
            text="🎧", 
            font=ctk.CTkFont(size=42)
        ).pack()
        
        ctk.CTkLabel(
            self.header,
            text="Urban Sound Classifier",
            font=ctk.CTkFont(family="Helvetica", size=26, weight="bold")
        ).pack()
        
        ctk.CTkLabel(
            self.header,
            text="AI-Powered Environmental Sound Recognition",
            font=ctk.CTkFont(size=13),
            text_color="gray60"
        ).pack(pady=(5, 0))
        
        # Content Area - Two Columns
        self.content_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, padx=30, pady=20, sticky="nsew")
        self.content_frame.grid_columnconfigure((0, 1), weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # === LEFT COLUMN: File Selection ===
        self.left_card = ctk.CTkFrame(
            self.content_frame, 
            fg_color=("gray90", "gray20"),
            corner_radius=20,
            border_width=2,
            border_color=("gray80", "gray30")
        )
        self.left_card.grid(row=0, column=0, padx=(0, 15), sticky="nsew")
        self.left_card.grid_columnconfigure(0, weight=1)
        self.left_card.grid_rowconfigure(2, weight=1)
        
        # Left Header
        self.left_header = ctk.CTkFrame(self.left_card, fg_color=("gray85", "gray25"), corner_radius=15, height=50)
        self.left_header.grid(row=0, column=0, padx=15, pady=15, sticky="ew")
        self.left_header.grid_propagate(False)
        
        ctk.CTkLabel(
            self.left_header,
            text="📁 Input",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("gray20", "gray90")
        ).place(relx=0.5, rely=0.5, anchor="center")
        
        # File Selection Content
        self.file_content = ctk.CTkFrame(self.left_card, fg_color="transparent")
        self.file_content.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.select_btn = ctk.CTkButton(
            self.file_content,
            text="Select Audio File",
            command=self.browse_file,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            corner_radius=12,
            fg_color="#6366f1",
            hover_color="#4f46e5",
            text_color="white",
            image=self.create_icon("📂", 20)
        )
        self.select_btn.pack(fill="x", pady=(10, 15))
        
        # File Info Box
        self.file_info = ctk.CTkFrame(
            self.file_content,
            fg_color=("gray95", "gray15"),
            corner_radius=12,
            height=80
        )
        self.file_info.pack(fill="x", pady=10)
        self.file_info.pack_propagate(False)
        
        self.file_icon_label = ctk.CTkLabel(
            self.file_info,
            text="🎵",
            font=ctk.CTkFont(size=28)
        )
        self.file_icon_label.pack(pady=(10, 5))
        
        self.file_name_label = ctk.CTkLabel(
            self.file_info,
            text="No file selected",
            font=ctk.CTkFont(size=12),
            text_color="gray50"
        )
        self.file_name_label.pack()
        
        # Waveform Visualizer
        self.wave_frame = ctk.CTkFrame(self.left_card, fg_color="transparent")
        self.wave_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        
        self.wave_canvas = ctk.CTkFrame(
            self.wave_frame,
            fg_color=("gray85", "gray25"),
            corner_radius=10,
            height=100
        )
        self.wave_canvas.pack(fill="both", expand=True)
        self.wave_canvas.pack_propagate(False)
        
        self.wave_visual = ctk.CTkLabel(
            self.wave_canvas,
            text="〰️ 〰️ 〰️ 〰️ 〰️",
            font=ctk.CTkFont(size=24),
            text_color="gray40"
        )
        self.wave_visual.place(relx=0.5, rely=0.5, anchor="center")
        
        # === RIGHT COLUMN: Detection & Results ===
        self.right_card = ctk.CTkFrame(
            self.content_frame,
            fg_color=("gray90", "gray20"),
            corner_radius=20,
            border_width=2,
            border_color=("gray80", "gray30")
        )
        self.right_card.grid(row=0, column=1, padx=(15, 0), sticky="nsew")
        self.right_card.grid_columnconfigure(0, weight=1)
        self.right_card.grid_rowconfigure(2, weight=1)
        
        # Right Header
        self.right_header = ctk.CTkFrame(self.right_card, fg_color=("gray85", "gray25"), corner_radius=15, height=50)
        self.right_header.grid(row=0, column=0, padx=15, pady=15, sticky="ew")
        self.right_header.grid_propagate(False)
        
        ctk.CTkLabel(
            self.right_header,
            text="🔍 Analysis",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("gray20", "gray90")
        ).place(relx=0.5, rely=0.5, anchor="center")
        
        # Detect Button
        self.detect_content = ctk.CTkFrame(self.right_card, fg_color="transparent")
        self.detect_content.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.detect_btn = ctk.CTkButton(
            self.detect_content,
            text="Analyze Sound",
            command=self.detect_sound,
            state="disabled",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            corner_radius=12,
            fg_color="gray40",
            hover_color="gray40",
            text_color="gray70",
            image=self.create_icon("🎯", 20)
        )
        self.detect_btn.pack(fill="x", pady=(10, 15))
        
        # Results Box
        self.result_box = ctk.CTkFrame(
            self.right_card,
            fg_color=("gray95", "gray15"),
            corner_radius=15
        )
        self.result_box.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.result_box.grid_columnconfigure(0, weight=1)
        self.result_box.grid_rowconfigure(1, weight=1)
        
        # Result Icon
        self.result_icon_frame = ctk.CTkFrame(self.result_box, fg_color="transparent", height=80)
        self.result_icon_frame.grid(row=0, column=0, pady=(20, 10))
        self.result_icon_frame.grid_propagate(False)
        
        self.result_icon = ctk.CTkLabel(
            self.result_icon_frame,
            text="🤔",
            font=ctk.CTkFont(size=48)
        )
        self.result_icon.place(relx=0.5, rely=0.5, anchor="center")
        
        # Result Text
        self.result_text_frame = ctk.CTkFrame(self.result_box, fg_color="transparent")
        self.result_text_frame.grid(row=1, column=0, sticky="nsew", padx=20)
        
        self.result_label = ctk.CTkLabel(
            self.result_text_frame,
            text="Ready to analyze",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="gray60"
        )
        self.result_label.pack()
        
        self.confidence_label = ctk.CTkLabel(
            self.result_text_frame,
            text="Select a file to begin",
            font=ctk.CTkFont(size=12),
            text_color="gray50"
        )
        self.confidence_label.pack(pady=(5, 0))
        
        # Confidence Bar
        self.confidence_bar_frame = ctk.CTkFrame(self.result_box, fg_color="transparent", height=30)
        self.confidence_bar_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))
        self.confidence_bar_frame.grid_columnconfigure(0, weight=1)
        self.confidence_bar_frame.grid_propagate(False)
        
        self.confidence_bar = ctk.CTkProgressBar(
            self.confidence_bar_frame,
            height=8,
            corner_radius=4,
            fg_color=("gray80", "gray30"),
            progress_color="#6366f1"
        )
        self.confidence_bar.grid(row=0, column=0, sticky="ew")
        self.confidence_bar.set(0)
        
        # Status Bar
        self.status_bar = ctk.CTkLabel(
            self.main_container,
            text="Ready • Model loaded",
            font=ctk.CTkFont(size=11),
            text_color="gray50"
        )
        self.status_bar.grid(row=2, column=0, padx=30, pady=(0, 20), sticky="w")
        
    def create_icon(self, emoji, size):
        # Helper to create icon-compatible strings
        return None  # CTkButton handles text+emoji natively
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.selected_file = file_path
            filename = os.path.basename(file_path)
            
            # Update left side
            self.file_name_label.configure(
                text=filename,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=("gray20", "gray90")
            )
            self.file_icon_label.configure(text="🔊")
            self.wave_visual.configure(
                text="█ ▄ ▂ ▄ █ ▂ ▄ █ ▂",
                text_color="#6366f1",
                font=ctk.CTkFont(size=20)
            )
            
            # Enable detect button
            self.detect_btn.configure(
                state="normal",
                fg_color="#10b981",  # Emerald
                hover_color="#059669",
                text_color="white",
                text="Analyze Sound"
            )
            
            # Reset right side
            self.result_icon.configure(text="🎯")
            self.result_label.configure(
                text="Ready to analyze",
                text_color="gray60"
            )
            self.confidence_label.configure(text="Click analyze to process")
            self.confidence_bar.set(0)
            self.confidence_bar.configure(progress_color="#6366f1")
            
            self.status_bar.configure(text=f"Loaded: {filename}")
            
    def detect_sound(self):
        if not self.selected_file or self.is_processing:
            return
            
        self.is_processing = True
        self.detect_btn.configure(
            state="disabled",
            text="Processing...",
            fg_color="gray40",
            text_color="gray70"
        )
        self.status_bar.configure(text="Extracting features...")
        self.update()
        
        try:
            input_data = preprocess_audio(self.selected_file)
            
            self.status_bar.configure(text="Running inference...")
            self.update()
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            idx = np.argmax(output_data)
            confidence = float(output_data[0][idx])
            label = LABELS[idx]
            icon = LABEL_ICONS.get(label, "🔊")
            
            # Animate results
            self.result_icon.configure(text="⏳")
            self.update()
            
            # Simulate processing steps
            for i in range(5):
                self.confidence_bar.set((i + 1) * 0.2)
                self.update()
                self.after(50)
            
            # Final result
            self.result_icon.configure(text=icon)
            self.result_label.configure(
                text=label,
                text_color=("gray10", "gray90")
            )
            self.confidence_label.configure(text=f"Confidence: {confidence*100:.1f}%")
            self.confidence_bar.set(confidence)
            
            # Color code
            if confidence > 0.8:
                color = "#10b981"  # Green
            elif confidence > 0.5:
                color = "#f59e0b"  # Orange
            else:
                color = "#ef4444"  # Red
                
            self.confidence_bar.configure(progress_color=color)
            self.status_bar.configure(text=f"Detected: {label} ({confidence*100:.1f}%)")
            
        except Exception as e:
            self.result_icon.configure(text="❌")
            self.result_label.configure(text="Error", text_color="#ef4444")
            self.confidence_label.configure(text=str(e)[:50])
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
            
        finally:
            self.is_processing = False
            self.detect_btn.configure(
                state="normal",
                text="Analyze Sound",
                fg_color="#10b981",
                hover_color="#059669",
                text_color="white"
            )

if __name__ == "__main__":
    app = ModernAudioClassifier()
    app.mainloop()