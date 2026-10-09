"""
ASL to Nigerian Languages Translator - Desktop GUI Application
Integrates dual-glove wireless telemetry, live dataset collection, model training,
real-time gesture inference, N-Atlas translation (Yoruba, Igbo, Hausa), and multilingual TTS.
"""
import os
import sys
import time
import threading
import logging
from typing import Dict, Any, Optional

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

import config
from receiver import DualGloveReceiver
from dataset_recorder import DatasetRecorder
from train_model import train_classifier
from gesture_recognizer import GestureRecognizer
from sentence_builder import SentenceBuilder
from translator import NAtlasTranslator
from tts_engine import TTSEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ASLTranslatorApp")

class ASLTranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("N-Atlas: ASL Glove Translator to Nigerian Languages")
        self.geometry("1100x800")
        self.minsize(980, 720)

        # Style configuration (Modern Dark Palette)
        self.configure(bg="#121820")
        self._init_styles()

        # Core Backend Modules
        self.receiver = DualGloveReceiver()
        self.recorder = DatasetRecorder(self.receiver)
        self.recognizer = GestureRecognizer()
        self.builder = SentenceBuilder()
        self.translator = NAtlasTranslator()
        self.tts = TTSEngine()

        # UI State Variables
        self.target_lang_var = tk.StringVar(value="Yoruba")
        self.neural_mode_var = tk.BooleanVar(value=False)
        self.auto_speak_var = tk.BooleanVar(value=False)
        self.sim_mode_var = tk.BooleanVar(value=False)
        self.live_recognition_active = tk.BooleanVar(value=True)

        # Build UI Layout
        self._build_header()
        self._build_connection_bar()
        self._build_tabs()

        # Start Receiver & Background GUI Refresh Loop
        self.receiver.start()
        self.after(100, self._telemetry_refresh_loop)

        # Handle clean exit
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_styles(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Dark theme palette
        bg_dark = "#121820"
        card_bg = "#1a2332"
        accent_blue = "#2563eb"
        text_light = "#f8fafc"
        text_dim = "#94a3b8"

        self.style.configure(".", background=bg_dark, foreground=text_light, font=("Segoe UI", 10))
        self.style.configure("TNotebook", background=bg_dark, borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#1e293b", foreground=text_dim, padding=[16, 8], font=("Segoe UI", 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", accent_blue)], foreground=[("selected", text_light)])
        self.style.configure("TFrame", background=bg_dark)
        self.style.configure("Card.TFrame", background=card_bg, relief="flat")
        self.style.configure("TLabel", background=bg_dark, foreground=text_light)
        self.style.configure("Card.TLabel", background=card_bg, foreground=text_light)
        self.style.configure("TProgressbar", thickness=14, troughcolor="#0f172a", background="#10b981")
        self.style.configure("TButton", padding=[10, 6], background="#1e293b", foreground=text_light, borderwidth=1)
        self.style.map("TButton", background=[("active", "#334155")])

    def _build_header(self):
        header_frame = tk.Frame(self, bg="#0f172a", height=70, padx=20, pady=10)
        header_frame.pack(fill=tk.X)

        title_label = tk.Label(
            header_frame,
            text="N-ATLAS : DUAL-GLOVE ASL TRANSLATOR",
            font=("Segoe UI", 14, "bold"),
            fg="#38bdf8",
            bg="#0f172a"
        )
        title_label.pack(side=tk.LEFT)

        subtitle_label = tk.Label(
            header_frame,
            text=" (ASL -> English -> Yorùbá / Igbo / Hausa)",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#0f172a"
        )
        subtitle_label.pack(side=tk.LEFT, padx=5)

        # Hardware Status indicators
        status_box = tk.Frame(header_frame, bg="#0f172a")
        status_box.pack(side=tk.RIGHT)

        # Simulation Toggle
        sim_chk = tk.Checkbutton(
            status_box,
            text="Simulation Mode",
            variable=self.sim_mode_var,
            command=self._toggle_simulation,
            bg="#0f172a",
            fg="#cbd5e1",
            selectcolor="#1e293b",
            activebackground="#0f172a",
            activeforeground="#38bdf8",
            font=("Segoe UI", 9)
        )
        sim_chk.pack(side=tk.LEFT, padx=12)

        # Left Hand Pill
        self.left_indicator = tk.Label(
            status_box,
            text="● LEFT GLOVE: OFF",
            font=("Segoe UI", 9, "bold"),
            fg="#ef4444",
            bg="#1e293b",
            padx=10,
            pady=4,
            relief="flat"
        )
        self.left_indicator.pack(side=tk.LEFT, padx=6)

        # Right Hand Pill
        self.right_indicator = tk.Label(
            status_box,
            text="● RIGHT GLOVE: OFF",
            font=("Segoe UI", 9, "bold"),
            fg="#ef4444",
            bg="#1e293b",
            padx=10,
            pady=4,
            relief="flat"
        )
        self.right_indicator.pack(side=tk.LEFT, padx=6)

    def _build_connection_bar(self):
        conn_frame = tk.Frame(self, bg="#161f2e", padx=16, pady=8, bd=1, relief="ridge")
        conn_frame.pack(fill=tk.X, padx=15, pady=(8, 0))

        # Title
        tk.Label(
            conn_frame,
            text="HARDWARE LINK & IP TESTER:",
            font=("Segoe UI", 9, "bold"),
            fg="#fbbf24",
            bg="#161f2e"
        ).pack(side=tk.LEFT, padx=(0, 10))

        # --- LEFT GLOVE SECTION ---
        l_box = tk.Frame(conn_frame, bg="#161f2e")
        l_box.pack(side=tk.LEFT, padx=8)

        # Inbuilt LED for Left Glove
        self.led_left = tk.Label(
            l_box,
            text="● INBUILT LED: OFF",
            font=("Segoe UI", 8, "bold"),
            fg="#64748b",
            bg="#0f172a",
            padx=7,
            pady=3,
            relief="groove"
        )
        self.led_left.pack(side=tk.LEFT, padx=4)

        tk.Label(l_box, text="Left IP:", font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#161f2e").pack(side=tk.LEFT, padx=(4, 2))
        self.entry_left_ip = tk.Entry(l_box, width=13, font=("Segoe UI", 9), bg="#0f172a", fg="#f8fafc", insertbackground="white")
        self.entry_left_ip.insert(0, "192.168.4.2")
        self.entry_left_ip.pack(side=tk.LEFT, padx=2)

        tk.Button(
            l_box,
            text="⚡ Ping",
            font=("Segoe UI", 8, "bold"),
            bg="#2563eb",
            fg="white",
            padx=6,
            pady=1,
            command=self._ping_left_glove
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            l_box,
            text="💡 Flash LED",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="white",
            padx=5,
            pady=1,
            command=self._blink_left_glove
        ).pack(side=tk.LEFT, padx=2)

        self.lbl_left_conn_status = tk.Label(
            l_box,
            text="Disconnected",
            font=("Segoe UI", 8),
            fg="#ef4444",
            bg="#161f2e"
        )
        self.lbl_left_conn_status.pack(side=tk.LEFT, padx=6)

        # Separator
        tk.Label(conn_frame, text="|", font=("Segoe UI", 12), fg="#334155", bg="#161f2e").pack(side=tk.LEFT, padx=8)

        # --- RIGHT GLOVE SECTION ---
        r_box = tk.Frame(conn_frame, bg="#161f2e")
        r_box.pack(side=tk.LEFT, padx=8)

        # Inbuilt LED for Right Glove
        self.led_right = tk.Label(
            r_box,
            text="● INBUILT LED: OFF",
            font=("Segoe UI", 8, "bold"),
            fg="#64748b",
            bg="#0f172a",
            padx=7,
            pady=3,
            relief="groove"
        )
        self.led_right.pack(side=tk.LEFT, padx=4)

        tk.Label(r_box, text="Right IP:", font=("Segoe UI", 9, "bold"), fg="#10b981", bg="#161f2e").pack(side=tk.LEFT, padx=(4, 2))
        self.entry_right_ip = tk.Entry(r_box, width=13, font=("Segoe UI", 9), bg="#0f172a", fg="#f8fafc", insertbackground="white")
        self.entry_right_ip.insert(0, "192.168.4.3")
        self.entry_right_ip.pack(side=tk.LEFT, padx=2)

        tk.Button(
            r_box,
            text="⚡ Ping",
            font=("Segoe UI", 8, "bold"),
            bg="#059669",
            fg="white",
            padx=6,
            pady=1,
            command=self._ping_right_glove
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            r_box,
            text="💡 Flash LED",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="white",
            padx=5,
            pady=1,
            command=self._blink_right_glove
        ).pack(side=tk.LEFT, padx=2)

        self.lbl_right_conn_status = tk.Label(
            r_box,
            text="Disconnected",
            font=("Segoe UI", 8),
            fg="#ef4444",
            bg="#161f2e"
        )
        self.lbl_right_conn_status.pack(side=tk.LEFT, padx=6)

    def _ping_left_glove(self):
        ip = self.entry_left_ip.get().strip()
        self.lbl_left_conn_status.config(text="Pinging...", fg="#f59e0b")
        def _worker():
            success, msg = self.receiver.ping_glove("LEFT", ip)
            color = "#10b981" if success else "#ef4444"
            self.after(0, lambda: self.lbl_left_conn_status.config(text=msg, fg=color))
        threading.Thread(target=_worker, daemon=True).start()

    def _blink_left_glove(self):
        ip = self.entry_left_ip.get().strip()
        self.receiver.blink_glove_led("LEFT", ip)

    def _ping_right_glove(self):
        ip = self.entry_right_ip.get().strip()
        self.lbl_right_conn_status.config(text="Pinging...", fg="#f59e0b")
        def _worker():
            success, msg = self.receiver.ping_glove("RIGHT", ip)
            color = "#10b981" if success else "#ef4444"
            self.after(0, lambda: self.lbl_right_conn_status.config(text=msg, fg=color))
        threading.Thread(target=_worker, daemon=True).start()

    def _blink_right_glove(self):
        ip = self.entry_right_ip.get().strip()
        self.receiver.blink_glove_led("RIGHT", ip)

    def _build_tabs(self):

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Tab 1: Live Sign Recognition & Multilingual Translation
        self.tab_translate = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_translate, text="  Live ASL Translation & TTS  ")
        self._build_translate_tab(self.tab_translate)

        # Tab 2: Glove Telemetry & Hardware Calibration
        self.tab_telemetry = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_telemetry, text="  Sensor Monitor & Calibration  ")
        self._build_telemetry_tab(self.tab_telemetry)

        # Tab 3: Dataset Studio
        self.tab_dataset = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dataset, text="  Dataset Studio  ")
        self._build_dataset_tab(self.tab_dataset)

        # Tab 4: Model Training
        self.tab_train = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="  Model Training & Evaluation  ")
        self._build_train_tab(self.tab_train)

    # =========================================================================
    # TAB 1: Live Translation & TTS
    # =========================================================================
    def _build_translate_tab(self, parent):
        grid = tk.Frame(parent, bg="#121820")
        grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top Section: Real-time detected sign and confidence
        top_card = tk.Frame(grid, bg="#1a2332", bd=1, relief="ridge", padx=15, pady=12)
        top_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top_card, text="LIVE ASL SIGN DETECTOR", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#1a2332").pack(anchor=tk.W)

        det_row = tk.Frame(top_card, bg="#1a2332")
        det_row.pack(fill=tk.X, pady=5)

        self.lbl_current_sign = tk.Label(det_row, text="WAITING FOR GESTURE...", font=("Segoe UI", 26, "bold"), fg="#38bdf8", bg="#1a2332")
        self.lbl_current_sign.pack(side=tk.LEFT)

        self.lbl_confidence = tk.Label(det_row, text="Confidence: 0.0%", font=("Segoe UI", 12), fg="#10b981", bg="#1a2332")
        self.lbl_confidence.pack(side=tk.RIGHT, padx=10)

        self.prob_bar = ttk.Progressbar(top_card, orient=tk.HORIZONTAL, length=400, mode="determinate")
        self.prob_bar.pack(fill=tk.X, pady=4)

        # Middle Section: English Sentence Construction
        mid_card = tk.Frame(grid, bg="#1a2332", bd=1, relief="ridge", padx=15, pady=12)
        mid_card.pack(fill=tk.X, pady=5)

        tk.Label(mid_card, text="CONSTRUCTED ENGLISH SENTENCE", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#1a2332").pack(anchor=tk.W)

        self.txt_english_sentence = tk.Text(mid_card, height=3, bg="#0f172a", fg="#f8fafc", font=("Segoe UI", 13), relief="flat", wrap=tk.WORD, padx=8, pady=8)
        self.txt_english_sentence.pack(fill=tk.X, pady=8)

        btn_row = tk.Frame(mid_card, bg="#1a2332")
        btn_row.pack(fill=tk.X)

        tk.Button(btn_row, text="⌫ Backspace", bg="#334155", fg="white", font=("Segoe UI", 9), command=self._action_backspace).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_row, text="✕ Clear Sentence", bg="#475569", fg="white", font=("Segoe UI", 9), command=self._action_clear_sentence).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_row, text="🔊 Speak English", bg="#2563eb", fg="white", font=("Segoe UI", 9, "bold"), command=self._speak_english).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_row, text="✓ Translate Now", bg="#059669", fg="white", font=("Segoe UI", 9, "bold"), command=self._trigger_translation).pack(side=tk.RIGHT, padx=3)

        # Bottom Section: N-Atlas Nigerian Languages Translation Card
        bot_card = tk.Frame(grid, bg="#1e293b", bd=2, relief="groove", padx=18, pady=14)
        bot_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        lang_control_row = tk.Frame(bot_card, bg="#1e293b")
        lang_control_row.pack(fill=tk.X, pady=(0, 10))

        tk.Label(lang_control_row, text="N-ATLAS NIGERIAN TRANSLATION", font=("Segoe UI", 11, "bold"), fg="#fbbf24", bg="#1e293b").pack(side=tk.LEFT)

        # Language Radio Buttons
        lang_pill_box = tk.Frame(lang_control_row, bg="#1e293b")
        lang_pill_box.pack(side=tk.RIGHT)

        for lang in ["Yoruba", "Igbo", "Hausa"]:
            r = tk.Radiobutton(
                lang_pill_box,
                text=lang,
                variable=self.target_lang_var,
                value=lang,
                command=self._trigger_translation,
                bg="#1e293b",
                fg="#f8fafc",
                selectcolor="#0f172a",
                activebackground="#1e293b",
                activeforeground="#38bdf8",
                font=("Segoe UI", 10, "bold")
            )
            r.pack(side=tk.LEFT, padx=8)

        # Translation Output Box
        self.lbl_translation_output = tk.Label(
            bot_card,
            text="[Translation will appear here]",
            font=("Segoe UI", 18, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
            wraplength=800,
            justify=tk.LEFT
        )
        self.lbl_translation_output.pack(anchor=tk.W, pady=10)

        # Phonetics Guide
        self.lbl_phonetics = tk.Label(
            bot_card,
            text="Pronunciation: -",
            font=("Segoe UI", 11, "italic"),
            fg="#94a3b8",
            bg="#1e293b"
        )
        self.lbl_phonetics.pack(anchor=tk.W, pady=(0, 10))

        # Bottom Actions & Engine Status
        bot_action_row = tk.Frame(bot_card, bg="#1e293b")
        bot_action_row.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_speak_nigerian = tk.Button(
            bot_action_row,
            text="🔊 Speak Translated Version (TTS)",
            font=("Segoe UI", 11, "bold"),
            bg="#d97706",
            fg="white",
            padx=16,
            pady=6,
            command=self._speak_nigerian
        )
        self.btn_speak_nigerian.pack(side=tk.LEFT)

        self.lbl_engine_info = tk.Label(
            bot_action_row,
            text="Engine: N-Atlas Lexicon | Diacritics: Active",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b"
        )
        self.lbl_engine_info.pack(side=tk.RIGHT, padx=10)

    # =========================================================================
    # TAB 2: Telemetry & Calibration
    # =========================================================================
    def _build_telemetry_tab(self, parent):
        container = tk.Frame(parent, bg="#121820")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        hands_box = tk.Frame(container, bg="#121820")
        hands_box.pack(fill=tk.BOTH, expand=True)

        # Left Hand Panel
        self.l_hand_frame = tk.LabelFrame(hands_box, text=" Left Hand Glove (5 Flex + MPU-9250) ", font=("Segoe UI", 10, "bold"), fg="#38bdf8", bg="#1a2332", padx=12, pady=10)
        self.l_hand_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Right Hand Panel
        self.r_hand_frame = tk.LabelFrame(hands_box, text=" Right Hand Glove (5 Flex + MPU-9250) ", font=("Segoe UI", 10, "bold"), fg="#10b981", bg="#1a2332", padx=12, pady=10)
        self.r_hand_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.l_bars = []
        self.l_bar_labels = []
        self.r_bars = []
        self.r_bar_labels = []

        # Left Hand Bars
        for i, fname in enumerate(finger_names):
            row = tk.Frame(self.l_hand_frame, bg="#1a2332")
            row.pack(fill=tk.X, pady=3)
            lbl = tk.Label(row, text=f"{fname}: 0%", width=12, anchor=tk.W, font=("Segoe UI", 9), bg="#1a2332", fg="#f8fafc")
            lbl.pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=180, mode="determinate")
            bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.l_bars.append(bar)
            self.l_bar_labels.append(lbl)

        self.l_imu_label = tk.Label(self.l_hand_frame, text="IMU: Ax:0.0 Ay:0.0 Az:1.0 | Gx:0 Gy:0 Gz:0", font=("Consolas", 8), fg="#94a3b8", bg="#1a2332", justify=tk.LEFT)
        self.l_imu_label.pack(fill=tk.X, pady=(12, 0))

        # Right Hand Bars
        for i, fname in enumerate(finger_names):
            row = tk.Frame(self.r_hand_frame, bg="#1a2332")
            row.pack(fill=tk.X, pady=3)
            lbl = tk.Label(row, text=f"{fname}: 0%", width=12, anchor=tk.W, font=("Segoe UI", 9), bg="#1a2332", fg="#f8fafc")
            lbl.pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=180, mode="determinate")
            bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.r_bars.append(bar)
            self.r_bar_labels.append(lbl)

        self.r_imu_label = tk.Label(self.r_hand_frame, text="IMU: Ax:0.0 Ay:0.0 Az:1.0 | Gx:0 Gy:0 Gz:0", font=("Consolas", 8), fg="#94a3b8", bg="#1a2332", justify=tk.LEFT)
        self.r_imu_label.pack(fill=tk.X, pady=(12, 0))

        # Calibration Controls Panel
        cal_panel = tk.LabelFrame(container, text=" Glove Min/Max ADC Calibration ", font=("Segoe UI", 10, "bold"), fg="#fbbf24", bg="#1e293b", padx=12, pady=10)
        cal_panel.pack(fill=tk.X, pady=(10, 0))

        cal_btns = tk.Frame(cal_panel, bg="#1e293b")
        cal_btns.pack(fill=tk.X)

        tk.Button(cal_btns, text="Set Flat Hand (0° Bend)", bg="#0284c7", fg="white", font=("Segoe UI", 9, "bold"), command=self._cal_set_flat).pack(side=tk.LEFT, padx=6)
        tk.Button(cal_btns, text="Set Closed Fist (90° Bend)", bg="#e11d48", fg="white", font=("Segoe UI", 9, "bold"), command=self._cal_set_bent).pack(side=tk.LEFT, padx=6)
        tk.Button(cal_btns, text="Reset Defaults", bg="#475569", fg="white", font=("Segoe UI", 9), command=self._cal_reset).pack(side=tk.LEFT, padx=6)

        self.lbl_cal_status = tk.Label(cal_btns, text="Status: Ready", fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 9))
        self.lbl_cal_status.pack(side=tk.RIGHT, padx=10)

    # =========================================================================
    # TAB 3: Dataset Studio
    # =========================================================================
    def _build_dataset_tab(self, parent):
        container = tk.Frame(parent, bg="#121820")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_controls = tk.Frame(container, bg="#1a2332", padx=15, pady=12, bd=1, relief="ridge")
        top_controls.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top_controls, text="Target ASL Gesture:", font=("Segoe UI", 10, "bold"), bg="#1a2332", fg="#f8fafc").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.cmb_gestures = ttk.Combobox(top_controls, values=config.DEFAULT_GESTURES, width=20, font=("Segoe UI", 10))
        self.cmb_gestures.set("HELLO")
        self.cmb_gestures.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(top_controls, text="Record Duration (sec):", font=("Segoe UI", 10), bg="#1a2332", fg="#f8fafc").grid(row=0, column=2, padx=10, pady=5)
        self.spn_duration = ttk.Spinbox(top_controls, from_=1.0, to=10.0, increment=0.5, width=6)
        self.spn_duration.set("3.0")
        self.spn_duration.grid(row=0, column=3, padx=5, pady=5)

        self.btn_record = tk.Button(top_controls, text="⏺ Record Sign Stream", bg="#e11d48", fg="white", font=("Segoe UI", 10, "bold"), padx=12, pady=4, command=self._start_recording_session)
        self.btn_record.grid(row=0, column=4, padx=15, pady=5)

        self.btn_snapshot = tk.Button(top_controls, text="+ Single Snapshot", bg="#334155", fg="white", font=("Segoe UI", 9), command=self._record_single_snapshot)
        self.btn_snapshot.grid(row=0, column=5, padx=5, pady=5)

        # Progress bar
        self.rec_prog_bar = ttk.Progressbar(top_controls, orient=tk.HORIZONTAL, mode="determinate")
        self.rec_prog_bar.grid(row=1, column=0, columnspan=6, sticky=tk.EW, pady=(10, 0))

        self.lbl_rec_status = tk.Label(top_controls, text="Status: Ready to capture", font=("Segoe UI", 9, "italic"), bg="#1a2332", fg="#94a3b8")
        self.lbl_rec_status.grid(row=2, column=0, columnspan=6, sticky=tk.W, pady=2)

        # Dataset Table & Stats
        bot_frame = tk.Frame(container, bg="#121820")
        bot_frame.pack(fill=tk.BOTH, expand=True)

        list_frame = tk.LabelFrame(bot_frame, text=" Dataset Summary (Class Distribution) ", font=("Segoe UI", 10, "bold"), fg="#38bdf8", bg="#1a2332", padx=10, pady=10)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.tree_dataset = ttk.Treeview(list_frame, columns=("Gesture", "Samples"), show="headings", height=12)
        self.tree_dataset.heading("Gesture", text="Gesture Class")
        self.tree_dataset.heading("Samples", text="Sample Count")
        self.tree_dataset.column("Gesture", width=180)
        self.tree_dataset.column("Samples", width=100)
        self.tree_dataset.pack(fill=tk.BOTH, expand=True)

        manage_col = tk.Frame(bot_frame, bg="#121820", width=220)
        manage_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        tk.Button(manage_col, text="🔄 Refresh Table", bg="#1e293b", fg="white", font=("Segoe UI", 9), width=22, command=self._refresh_dataset_table).pack(pady=4)
        tk.Button(manage_col, text="⚡ Generate Bootstrap Data", bg="#0284c7", fg="white", font=("Segoe UI", 9, "bold"), width=22, command=self._generate_bootstrap).pack(pady=4)
        tk.Button(manage_col, text="🗑 Delete Selected Gesture", bg="#b91c1c", fg="white", font=("Segoe UI", 9), width=22, command=self._delete_selected_gesture).pack(pady=4)
        tk.Button(manage_col, text="⚠ Wipe Entire Dataset", bg="#7f1d1d", fg="white", font=("Segoe UI", 9), width=22, command=self._wipe_dataset).pack(pady=4)

        self._refresh_dataset_table()

    # =========================================================================
    # TAB 4: Model Training
    # =========================================================================
    def _build_train_tab(self, parent):
        container = tk.Frame(parent, bg="#121820")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        card = tk.Frame(container, bg="#1a2332", padx=20, pady=16, bd=1, relief="ridge")
        card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(card, text="CLASSIFIER CONFIGURATION", font=("Segoe UI", 11, "bold"), fg="#38bdf8", bg="#1a2332").pack(anchor=tk.W, pady=(0, 10))

        opt_row = tk.Frame(card, bg="#1a2332")
        opt_row.pack(fill=tk.X, pady=5)

        tk.Label(opt_row, text="Algorithm:", font=("Segoe UI", 10), bg="#1a2332", fg="#f8fafc").pack(side=tk.LEFT)
        self.model_type_var = tk.StringVar(value="rf")
        tk.Radiobutton(opt_row, text="Random Forest", variable=self.model_type_var, value="rf", bg="#1a2332", fg="white", selectcolor="#0f172a").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(opt_row, text="MLP Neural Net", variable=self.model_type_var, value="mlp", bg="#1a2332", fg="white", selectcolor="#0f172a").pack(side=tk.LEFT, padx=10)

        self.btn_train = tk.Button(
            card,
            text="🚀 Train ASL Recognition Model",
            bg="#2563eb",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=16,
            pady=8,
            command=self._start_training
        )
        self.btn_train.pack(anchor=tk.W, pady=12)

        # Metrics Display Box
        metrics_box = tk.LabelFrame(container, text=" Training Evaluation Results ", font=("Segoe UI", 10, "bold"), fg="#10b981", bg="#1a2332", padx=15, pady=12)
        metrics_box.pack(fill=tk.BOTH, expand=True)

        self.lbl_train_acc = tk.Label(metrics_box, text="Test Accuracy: Not Trained Yet", font=("Segoe UI", 14, "bold"), fg="#10b981", bg="#1a2332")
        self.lbl_train_acc.pack(anchor=tk.W, pady=4)

        self.txt_train_report = tk.Text(metrics_box, bg="#0f172a", fg="#f8fafc", font=("Consolas", 10), relief="flat")
        self.txt_train_report.pack(fill=tk.BOTH, expand=True, pady=6)

    # =========================================================================
    # Background Refresh & Inference Loop
    # =========================================================================
    def _telemetry_refresh_loop(self):
        try:
            l_state, r_state = self.receiver.get_latest_states()

            # Update Header Badges & Connection Bar Inbuilt LEDs
            if l_state["connected"]:
                self.left_indicator.config(text=f"● LEFT GLOVE: {self.receiver.left_fps:.0f} Hz", fg="#10b981")
                self.led_left.config(text="● INBUILT LED: ON", fg="#38bdf8", bg="#032b43")
                self.lbl_left_conn_status.config(text=f"Active ({self.receiver.left_fps:.0f} Hz)", fg="#10b981")
                if l_state.get("ip_addr") and self.entry_left_ip.get().strip() in ["", "192.168.4.2"]:
                    self.entry_left_ip.delete(0, tk.END)
                    self.entry_left_ip.insert(0, l_state["ip_addr"])
            else:
                self.left_indicator.config(text="● LEFT GLOVE: OFF", fg="#ef4444")
                self.led_left.config(text="● INBUILT LED: OFF", fg="#64748b", bg="#0f172a")
                if "Pinging" not in self.lbl_left_conn_status.cget("text"):
                    self.lbl_left_conn_status.config(text="Disconnected", fg="#ef4444")

            if r_state["connected"]:
                self.right_indicator.config(text=f"● RIGHT GLOVE: {self.receiver.right_fps:.0f} Hz", fg="#10b981")
                self.led_right.config(text="● INBUILT LED: ON", fg="#10b981", bg="#064e3b")
                self.lbl_right_conn_status.config(text=f"Active ({self.receiver.right_fps:.0f} Hz)", fg="#10b981")
                if r_state.get("ip_addr") and self.entry_right_ip.get().strip() in ["", "192.168.4.3"]:
                    self.entry_right_ip.delete(0, tk.END)
                    self.entry_right_ip.insert(0, r_state["ip_addr"])
            else:
                self.right_indicator.config(text="● RIGHT GLOVE: OFF", fg="#ef4444")
                self.led_right.config(text="● INBUILT LED: OFF", fg="#64748b", bg="#0f172a")
                if "Pinging" not in self.lbl_right_conn_status.cget("text"):
                    self.lbl_right_conn_status.config(text="Disconnected", fg="#ef4444")

            # Update Telemetry Bars
            for i in range(5):
                l_val = int(l_state["flex_norm"][i] * 100)
                self.l_bars[i]["value"] = l_val
                self.l_bar_labels[i].config(text=f"Finger {i+1}: {l_val}%")

                r_val = int(r_state["flex_norm"][i] * 100)
                self.r_bars[i]["value"] = r_val
                self.r_bar_labels[i].config(text=f"Finger {i+1}: {r_val}%")

            lax, lay, laz = l_state["accel"]
            lgx, lgy, lgz = l_state["gyro"]
            self.l_imu_label.config(text=f"Ax:{lax} Ay:{lay} Az:{laz} | Gx:{lgx} Gy:{lgy} Gz:{lgz}")

            rax, ray, raz = r_state["accel"]
            rgx, rgy, rgz = r_state["gyro"]
            self.r_imu_label.config(text=f"Ax:{rax} Ay:{ray} Az:{raz} | Gx:{rgx} Gy:{rgy} Gz:{rgz}")

            # Real-Time Gesture Inference (Gated: only runs when gloves are connected or in simulation)
            is_glove_active = l_state["connected"] or r_state["connected"] or self.receiver.simulation_mode

            if self.live_recognition_active.get() and self.recognizer.is_loaded and is_glove_active:
                vec = self.receiver.get_synchronized_feature_vector()
                label, conf, is_new_event = self.recognizer.process_live_stream(vec)

                if label and label not in ["IDLE", "REST"]:
                    self.lbl_current_sign.config(text=f"SIGN: {label}")
                    self.lbl_confidence.config(text=f"Confidence: {conf*100:.1f}%")
                    self.prob_bar["value"] = int(conf * 100)
                elif label in ["IDLE", "REST"]:
                    self.lbl_current_sign.config(text="RESTING (IDLE)")
                    self.lbl_confidence.config(text=f"Confidence: {conf*100:.1f}%")
                    self.prob_bar["value"] = int(conf * 100)
                else:
                    self.lbl_current_sign.config(text="LISTENING...")
                    self.lbl_confidence.config(text="Confidence: 0.0%")
                    self.prob_bar["value"] = 0

                if is_new_event and label and label not in ["IDLE", "REST"]:
                    self._on_gesture_event(label)
            else:
                # Gloves are not streaming or recognition paused
                if not is_glove_active:
                    self.lbl_current_sign.config(text="WAITING FOR GLOVES (DISCONNECTED)")
                else:
                    self.lbl_current_sign.config(text="PAUSED")
                self.lbl_confidence.config(text="Confidence: 0.0%")
                self.prob_bar["value"] = 0
                self.recognizer.reset_state()


        except Exception as e:
            logger.debug(f"Refresh loop exception: {e}")

        self.after(50, self._telemetry_refresh_loop) # ~20Hz UI refresh

    def _on_gesture_event(self, sign: str):
        """Called when a stable gesture is confirmed."""
        sentence = self.builder.add_token(sign)
        self.txt_english_sentence.delete("1.0", tk.END)
        self.txt_english_sentence.insert("1.0", sentence)
        self._trigger_translation()

    # =========================================================================
    # Translation & TTS Handlers
    # =========================================================================
    def _trigger_translation(self):
        eng_text = self.txt_english_sentence.get("1.0", tk.END).strip()
        if not eng_text:
            return

        target_lang = self.target_lang_var.get()
        result = self.translator.translate(eng_text, target_lang)

        trans_text = result["translation"]
        phonetic = result["phonetic"]
        engine = result["engine"]

        self.lbl_translation_output.config(text=trans_text)
        self.lbl_phonetics.config(text=f"Pronunciation: {phonetic}")
        self.lbl_engine_info.config(text=f"Engine: {engine} | Target: {target_lang}")

        if self.auto_speak_var.get():
            self._speak_nigerian()

    def _speak_english(self):
        text = self.txt_english_sentence.get("1.0", tk.END).strip()
        if text:
            self.tts.speak(text, "English")

    def _speak_nigerian(self):
        text = self.lbl_translation_output.cget("text")
        if text and text != "[Translation will appear here]":
            self.tts.speak(text, self.target_lang_var.get())

    def _action_backspace(self):
        new_text = self.builder.backspace()
        self.txt_english_sentence.delete("1.0", tk.END)
        self.txt_english_sentence.insert("1.0", new_text)
        self._trigger_translation()

    def _action_clear_sentence(self):
        self.builder.clear()
        self.txt_english_sentence.delete("1.0", tk.END)
        self.lbl_translation_output.config(text="[Translation will appear here]")
        self.lbl_phonetics.config(text="Pronunciation: -")

    # =========================================================================
    # Calibration Handlers
    # =========================================================================
    def _cal_set_flat(self):
        l_s, r_s = self.receiver.get_latest_states()
        cal = self.receiver.calibration
        cal["LEFT"]["flex_flat"] = list(l_s["flex_raw"])
        cal["RIGHT"]["flex_flat"] = list(r_s["flex_raw"])
        self.receiver.save_calibration(cal)
        self.lbl_cal_status.config(text="Status: Flat Hand (0°) Calibrated!", fg="#10b981")

    def _cal_set_bent(self):
        l_s, r_s = self.receiver.get_latest_states()
        cal = self.receiver.calibration
        cal["LEFT"]["flex_bent"] = list(l_s["flex_raw"])
        cal["RIGHT"]["flex_bent"] = list(r_s["flex_raw"])
        self.receiver.save_calibration(cal)
        self.lbl_cal_status.config(text="Status: Closed Fist (90°) Calibrated!", fg="#10b981")

    def _cal_reset(self):
        cal = self.receiver.load_default_calibration()
        self.receiver.save_calibration(cal)
        self.lbl_cal_status.config(text="Status: Defaults Restored", fg="#94a3b8")

    # =========================================================================
    # Dataset & Training Handlers
    # =========================================================================
    def _toggle_simulation(self):
        if self.sim_mode_var.get():
            self.receiver.start_simulation()
        else:
            self.receiver.stop_simulation()

    def _start_recording_session(self):
        gesture = self.cmb_gestures.get().strip().upper()
        if not gesture:
            messagebox.showwarning("Warning", "Select or type a gesture label first.")
            return

        try:
            dur = float(self.spn_duration.get())
        except ValueError:
            dur = 3.0

        self.btn_record.config(state=tk.DISABLED)
        self.lbl_rec_status.config(text=f"Recording '{gesture}' in progress...", fg="#f59e0b")

        def _prog(progress, count):
            self.rec_prog_bar["value"] = int(progress * 100)

        def _done(total):
            self.btn_record.config(state=tk.NORMAL)
            self.lbl_rec_status.config(text=f"Finished! Saved {total} samples for '{gesture}'.", fg="#10b981")
            self._refresh_dataset_table()

        self.recorder.record_gesture_session(gesture, duration_sec=dur, progress_cb=_prog, done_cb=_done)

    def _record_single_snapshot(self):
        gesture = self.cmb_gestures.get().strip().upper()
        if not gesture:
            return
        vec = self.receiver.get_synchronized_feature_vector()
        self.recorder.append_single_frame(gesture, vec)
        self._refresh_dataset_table()
        self.lbl_rec_status.config(text=f"Snapshot saved for '{gesture}'", fg="#10b981")

    def _refresh_dataset_table(self):
        for item in self.tree_dataset.get_children():
            self.tree_dataset.delete(item)
        summary = self.recorder.get_dataset_summary()
        for gest, count in sorted(summary.items()):
            self.tree_dataset.insert("", tk.END, values=(gest, count))

    def _generate_bootstrap(self):
        self.recorder.generate_synthetic_bootstrap_dataset()
        self._refresh_dataset_table()
        messagebox.showinfo("Success", "Synthetic dataset generated for all common ASL signs!")

    def _delete_selected_gesture(self):
        sel = self.tree_dataset.selection()
        if not sel:
            return
        item = self.tree_dataset.item(sel[0])
        gest = item["values"][0]
        if messagebox.askyesno("Confirm", f"Delete all samples for gesture '{gest}'?"):
            self.recorder.delete_gesture_samples(gest)
            self._refresh_dataset_table()

    def _wipe_dataset(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to delete the ENTIRE dataset?"):
            self.recorder.clear_dataset()
            self._refresh_dataset_table()

    def _start_training(self):
        self.btn_train.config(state=tk.DISABLED, text="Training in progress...")

        def _worker():
            success, metrics = train_classifier(model_type=self.model_type_var.get())

            def _update_ui():
                self.btn_train.config(state=tk.NORMAL, text="🚀 Train ASL Recognition Model")
                if success:
                    acc = metrics["accuracy"] * 100
                    self.lbl_train_acc.config(text=f"Test Accuracy: {acc:.2f}% (Classes: {metrics['num_classes']})", fg="#10b981")
                    self.txt_train_report.delete("1.0", tk.END)
                    report_str = f"Model: {metrics['model_type'].upper()}\n"
                    report_str += f"Training Samples: {metrics['train_samples']} | Test Samples: {metrics['test_samples']}\n\n"
                    report_str += "Classes: " + ", ".join(metrics["classes"]) + "\n"
                    self.txt_train_report.insert("1.0", report_str)

                    # Reload recognizer model
                    self.recognizer.load_model()
                    messagebox.showinfo("Trained!", f"Model successfully trained with {acc:.1f}% accuracy and loaded for live recognition!")
                else:
                    self.lbl_train_acc.config(text="Training Failed", fg="#ef4444")
                    messagebox.showerror("Error", metrics.get("error", "Unknown error"))

            self.after(0, _update_ui)

        threading.Thread(target=_worker, daemon=True).start()

    def on_close(self):
        self.receiver.stop()
        self.destroy()

if __name__ == "__main__":
    app = ASLTranslatorApp()
    app.mainloop()
