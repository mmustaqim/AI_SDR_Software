#!/usr/bin/env python3
"""
HackRF Wideband Sweep (10 MHz – 6 GHz) with Waterfall + Band Classification + CSV Logging
"""

import sys, time, collections, csv, os
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from gnuradio import gr, blocks
import osmosdr
from scipy.signal import find_peaks

# ---------------- Config ----------------
START_FREQ = 10e6
STOP_FREQ = 6e9
STEP_SIZE = 20e6       # HackRF usable ~20 MHz BW
SAMPLE_RATE = 20e6
FFT_SIZE = 1024        # smaller FFT to reduce CPU load
DWELL_TIME = 0.4       # seconds per step
DEVICE_ARGS = "hackrf=0"
POLL_INTERVAL_MS = 500 # update GUI every 0.5s
WATERFALL_HEIGHT = 200 # rows stored in waterfall
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
    def __init__(self, samp_rate, fft_size):
        gr.top_block.__init__(self)
        self.src = osmosdr.source(args=DEVICE_ARGS)
        self.src.set_sample_rate(samp_rate)
        self.src.set_center_freq(100e6)
        self.src.set_gain(40)

        self.stream_to_vec = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size)
        self.vec_sink = blocks.vector_sink_c(fft_size)
        self.connect(self.src, self.stream_to_vec, self.vec_sink)

    def set_center_freq(self, f):
        self.src.set_center_freq(f)

    def get_data(self):
        data = self.vec_sink.data()
        if len(data) < FFT_SIZE:
            return None
        return np.array(data[-FFT_SIZE:], dtype=np.complex64)

class SweepGUI(QtWidgets.QMainWindow):
    def __init__(self, tb, sr, fft_size):
        super().__init__()
        self.tb = tb
        self.sr = sr
        self.fft_size = fft_size
        self.current_freq = START_FREQ
        self.last_step_time = 0

        # store waterfall
        self.waterfall = collections.deque(maxlen=WATERFALL_HEIGHT)

        # detection log
        self.log_entries = collections.deque(maxlen=100)

        # prepare CSV
        self.csv_file = open(CSV_FILE, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "Frequency_MHz", "Power_dB", "Band"])

        self.setWindowTitle("HackRF Wideband Sweep with Waterfall + CSV Logging")
        self.resize(1000, 700)
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(); cw.setLayout(v)

        # waterfall plot
        self.img = pg.ImageView(view=pg.PlotItem())
        v.addWidget(self.img, 3)

        # log box
        self.text = QtWidgets.QTextEdit(); self.text.setReadOnly(True)
        v.addWidget(self.text, 1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(POLL_INTERVAL_MS)

    def update_spectrum(self):
        # step sweep
        if time.time() - self.last_step_time > DWELL_TIME:
            self.current_freq += STEP_SIZE
            if self.current_freq > STOP_FREQ:
                self.current_freq = START_FREQ
            self.tb.set_center_freq(self.current_freq)
            self.last_step_time = time.time()

        raw = self.tb.get_data()
        if raw is None:
            return

        win = np.hanning(self.fft_size)
        spec = np.fft.fftshift(np.fft.fft(raw*win, n=self.fft_size))
        psd = 20*np.log10(np.abs(spec)+1e-12)
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sr)) + self.current_freq

        # add to waterfall
        self.waterfall.append(psd)
        img_data = np.array(self.waterfall)
        self.img.setImage(img_data, autoLevels=True)

        # detect peaks
        peaks, _ = find_peaks(psd, prominence=10)
        for p in peaks:
            f = freqs[p]
            band = classify_frequency(f)
            if band != "Unknown":
                entry = f"{f/1e6:.2f} MHz → {band}"
                self.log_entries.append(entry)
                self.csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                          f/1e6, float(psd[p]), band])
                self.csv_file.flush()

        # update GUI log
        self.text.setPlainText("\n".join(list(self.log_entries)[-20:]))

    def closeEvent(self, event):
        """Ensure CSV file is closed properly on exit."""
        self.csv_file.close()
        event.accept()

def main():
    tb = GrTop(SAMPLE_RATE, FFT_SIZE)
    tb.start()
    app = QtWidgets.QApplication(sys.argv)
    win = SweepGUI(tb, SAMPLE_RATE, FFT_SIZE)
    win.show()
    rc = app.exec_()
    tb.stop(); tb.wait()
    sys.exit(rc)

if __name__ == "__main__":
    main()
