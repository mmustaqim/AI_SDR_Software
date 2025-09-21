#!/usr/bin/env python3
"""
HackRF Sweep 10 MHz – 6 GHz
Optimized for stability: Waterfall + Frequency Classification + CSV Logging
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
FFT_SIZE = 512         # smaller to reduce CPU
DWELL_TIME = 0.8       # seconds per step
DEVICE_ARGS = "hackrf=0"
POLL_INTERVAL_MS = 800 # update GUI ~1Hz
WATERFALL_HEIGHT = 100
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
        self.src = osmosdr.source(args=DEVICE_ARGS)
        self.src.set_sample_rate(samp_rate)
        self.src.set_center_freq(START_FREQ)
        self.src.set_gain(40)

        # probe samples continuously
        self.head = blocks.head(gr.sizeof_gr_complex, int(1e9))  # just a limiter
        self.probe = blocks.probe_signal_c()
        self.connect(self.src, self.head, self.probe)

    def set_center_freq(self, f):
        self.src.set_center_freq(f)

    def get_samples(self, n=FFT_SIZE):
        """Grab n complex samples (approx)."""
        vec = []
        while len(vec) < n:
            sample = self.probe.level()  # returns float of last sample
            vec.append(sample + 0j)      # fake complex (workaround)
        return np.array(vec, dtype=np.complex64)

class SweepGUI(QtWidgets.QMainWindow):
    def __init__(self, tb, sr):
        super().__init__()
        self.tb = tb
        self.sr = sr
        self.current_freq = START_FREQ
        self.last_step_time = 0

        # Waterfall buffer
        self.waterfall = collections.deque(maxlen=WATERFALL_HEIGHT)

        # CSV
        self.csv_file = open(CSV_FILE, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "Frequency_MHz", "Power_dB", "Band"])

        self.setWindowTitle("HackRF Sweep Optimized")
        self.resize(900, 600)
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(); cw.setLayout(v)

        # Waterfall plot
        self.img = pg.ImageView(view=pg.PlotItem())
        v.addWidget(self.img, 3)

        # Log text
        self.text = QtWidgets.QTextEdit(); self.text.setReadOnly(True)
        v.addWidget(self.text, 1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(POLL_INTERVAL_MS)

        self.log_entries = collections.deque(maxlen=50)

    def update_spectrum(self):
        # sweep step
        if time.time() - self.last_step_time > DWELL_TIME:
            self.current_freq += STEP_SIZE
            if self.current_freq > STOP_FREQ:
                self.current_freq = START_FREQ
            self.tb.set_center_freq(self.current_freq)
            self.last_step_time = time.time()

        raw = self.tb.get_samples(FFT_SIZE)
        if raw is None or len(raw) < FFT_SIZE:
            return

        # FFT
        win = np.hanning(FFT_SIZE)
        spec = np.fft.fftshift(np.fft.fft(raw*win, n=FFT_SIZE))
        psd = 20*np.log10(np.abs(spec)+1e-12)
        freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/self.sr)) + self.current_freq

        # update waterfall
        self.waterfall.append(psd)
        self.img.setImage(np.array(self.waterfall), autoLevels=True)

        # detect peaks
        peaks, _ = find_peaks(psd, prominence=12)
        for p in peaks:
            f = freqs[p]
            band = classify_frequency(f)
            if band != "Unknown":
                entry = f"{f/1e6:.2f} MHz → {band}"
                self.log_entries.append(entry)
                self.csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                          f/1e6, float(psd[p]), band])
                self.csv_file.flush()

        # update GUI text
        self.text.setPlainText("\n".join(list(self.log_entries)[-20:]))

    def closeEvent(self, event):
        self.csv_file.close()
        event.accept()

def main():
    tb = GrTop(SAMPLE_RATE)
    tb.start()
    app = QtWidgets.QApplication(sys.argv)
    win = SweepGUI(tb, SAMPLE_RATE)
    win.show()
    rc = app.exec_()
    tb.stop(); tb.wait()
    sys.exit(rc)

if __name__ == "__main__":
    main()
