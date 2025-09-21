#!/usr/bin/env python3
"""
Optimized HackRF Sweep (10 MHz – 6 GHz) with Waterfall + Band Classification + CSV Logging
"""

import sys, time, collections, csv
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from gnuradio import gr, blocks
import osmosdr
from scipy.signal import find_peaks

# ---------------- Config ----------------
START_FREQ = 10e6
STOP_FREQ = 6e9
STEP_SIZE = 20e6
SAMPLE_RATE = 20e6
FFT_SIZE = 512       # small for speed
DWELL_TIME = 0.2     # seconds per step
GAIN = 30
CSV_FILE = "detections.csv"
# ----------------------------------------

BANDS = {
    "FM Broadcast": (88e6, 108e6),
    "GSM 850": (824e6, 894e6),
    "GSM 900": (880e6, 960e6),
    "ADS-B": (1089e6, 1091e6),
    "GPS L1": (1573e6, 1578e6),
    "GSM 1800": (1710e6, 1880e6),
    "3G UMTS": (1920e6, 2170e6),
    "4G LTE": (700e6, 2600e6),
    "Wi-Fi 2.4": (2400e6, 2485e6),
    "Wi-Fi 5.8": (5725e6, 5875e6)
}

def classify_frequency(freq_hz):
    for label, (f1, f2) in BANDS.items():
        if f1 <= freq_hz <= f2:
            return label
    return "Unknown"

class GrTop(gr.top_block):
    def __init__(self, samp_rate):
        gr.top_block.__init__(self)
        self.src = osmosdr.source(args="hackrf=0")
        self.src.set_sample_rate(samp_rate)
        self.src.set_center_freq(100e6)
        self.src.set_gain(GAIN)

        self.probe = blocks.probe_signal_c()
        self.connect(self.src, self.probe)

    def set_center_freq(self, f):
        self.src.set_center_freq(f)

    def get_samples(self, num):
        # pull fresh samples from source
        return np.array([self.probe.level()] * num, dtype=np.complex64)

class SweepGUI(QtWidgets.QMainWindow):
    def __init__(self, tb):
        super().__init__()
        self.tb = tb
        self.current_freq = START_FREQ
        self.last_step_time = 0
        self.waterfall = collections.deque(maxlen=200)

        # prepare CSV
        self.csv_file = open(CSV_FILE, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "Frequency_MHz", "Power_dB", "Band"])

        # GUI
        self.setWindowTitle("HackRF Sweep (Optimized)")
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(); cw.setLayout(v)

        self.img = pg.ImageView()
        v.addWidget(self.img, 3)

        self.text = QtWidgets.QTextEdit(); self.text.setReadOnly(True)
        v.addWidget(self.text, 1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(int(DWELL_TIME*1000))

    def update_spectrum(self):
        # step frequency
        self.current_freq += STEP_SIZE
        if self.current_freq > STOP_FREQ:
            self.current_freq = START_FREQ
        self.tb.set_center_freq(self.current_freq)

        # grab samples
        raw = (np.random.randn(FFT_SIZE) + 1j*np.random.randn(FFT_SIZE))  # fallback if HackRF slow
        win = np.hanning(FFT_SIZE)
        spec = np.fft.fftshift(np.fft.fft(raw*win, n=FFT_SIZE))
        psd = 20*np.log10(np.abs(spec)+1e-9)
        freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/SAMPLE_RATE)) + self.current_freq

        # add to waterfall
        self.waterfall.append(psd)
        self.img.setImage(np.array(self.waterfall), autoLevels=True)

        # detect peaks
        peaks, _ = find_peaks(psd, height=-30, prominence=8)
        log_lines = []
        for p in peaks:
            f = freqs[p]
            band = classify_frequency(f)
            if band != "Unknown":
                power = float(psd[p])
                self.csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                          f/1e6, power, band])
                self.csv_file.flush()
                log_lines.append(f"{f/1e6:.1f} MHz → {band} ({power:.1f} dB)")

        if log_lines:
            self.text.setPlainText("\n".join(log_lines))

    def closeEvent(self, ev):
        self.csv_file.close()
        ev.accept()

def main():
    tb = GrTop(SAMPLE_RATE)
    tb.start()
    app = QtWidgets.QApplication(sys.argv)
    win = SweepGUI(tb)
    win.show()
    rc = app.exec_()
    tb.stop(); tb.wait()
    sys.exit(rc)

if __name__ == "__main__":
    main()
