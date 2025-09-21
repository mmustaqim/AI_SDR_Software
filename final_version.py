#!/usr/bin/env python3
"""
HackRF Sweep 10 MHz – 6 GHz
- Waterfall with GQRX-style colormap
- Accurate Band Classification
- CSV Logging (Frequency, Power dB, Band)
- Dropdown to jump directly to bands (Wi-Fi, ADS-B, GPS, etc.)
"""

import sys, time, collections, csv
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from gnuradio import gr, blocks
import osmosdr
from scipy.signal import find_peaks

# ---------------- Config ----------------
FULL_START = 10e6
FULL_STOP = 6e9
STEP_SIZE = 20e6
SAMPLE_RATE = 20e6
FFT_SIZE = 512
DWELL_TIME = 0.3
DEVICE_ARGS = "hackrf=0"
POLL_INTERVAL_MS = 500
WATERFALL_HEIGHT = 400
CSV_FILE = "detections.csv"
# ----------------------------------------

BANDS = {
    "FM Broadcast": (88e6, 108e6),
    "GSM 850": (824e6, 894e6),
    "GSM 900": (880e6, 960e6),
    "ADS-B": (1089e6, 1091e6),
    "GPS L1": (1574e6, 1576e6),
    "GSM 1800": (1710e6, 1880e6),
    "3G UMTS": (1920e6, 2170e6),
    "4G LTE 700": (699e6, 801e6),
    "4G LTE 2100": (2110e6, 2170e6),
    "Wi-Fi 2.4": (2400e6, 2485e6),
    "Wi-Fi 5.8": (5725e6, 5875e6),
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
        self.src.set_center_freq(FULL_START)
        self.src.set_gain(40)

        # vector sink for FFT data
        self.stream_to_vec = blocks.stream_to_vector(gr.sizeof_gr_complex, FFT_SIZE)
        self.vec_sink = blocks.vector_sink_c(FFT_SIZE)
        self.connect(self.src, self.stream_to_vec, self.vec_sink)

    def set_center_freq(self, f):
        self.src.set_center_freq(f)

    def get_data(self):
        data = self.vec_sink.data()
        if len(data) < FFT_SIZE:
            return None
        self.vec_sink.reset()
        return np.array(data[-FFT_SIZE:], dtype=np.complex64)

class SweepGUI(QtWidgets.QMainWindow):
    def __init__(self, tb, sr):
        super().__init__()
        self.tb = tb
        self.sr = sr

        # sweep range (default full sweep)
        self.start_freq = FULL_START
        self.stop_freq = FULL_STOP
        self.current_freq = self.start_freq
        self.last_step_time = 0

        self.waterfall = collections.deque(maxlen=WATERFALL_HEIGHT)

        # CSV logging
        self.csv_file = open(CSV_FILE, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "Frequency_MHz", "Power_dB", "Band"])

        # window setup
        self.setWindowTitle("HackRF Sweep – Band Jump Enabled")
        self.resize(1400, 700)
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(); cw.setLayout(v)

        # Band selection dropdown
        self.combo = QtWidgets.QComboBox()
        self.combo.addItem("Full Sweep")
        for band in BANDS.keys():
            self.combo.addItem(band)
        self.combo.currentTextChanged.connect(self.change_band)
        v.addWidget(self.combo)

        # Waterfall (GQRX-style)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.img = pg.ImageView()
        self.img.ui.roiBtn.hide()
        self.img.ui.menuBtn.hide()
        self.img.setColorMap(pg.colormap.get('inferno'))
        v.addWidget(self.img, 3)

        # Log text
        self.text = QtWidgets.QTextEdit(); self.text.setReadOnly(True)
        v.addWidget(self.text, 1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_spectrum)
        self.timer.start(POLL_INTERVAL_MS)

        self.log_entries = collections.deque(maxlen=50)

    def change_band(self, band_name):
        """Adjust sweep range based on selection"""
        if band_name == "Full Sweep":
            self.start_freq, self.stop_freq = FULL_START, FULL_STOP
        else:
            self.start_freq, self.stop_freq = BANDS[band_name]
        self.current_freq = self.start_freq
        self.tb.set_center_freq(self.current_freq)

    def update_spectrum(self):
        # step sweep
        if time.time() - self.last_step_time > DWELL_TIME:
            self.current_freq += STEP_SIZE
            if self.current_freq > self.stop_freq:
                self.current_freq = self.start_freq
            self.tb.set_center_freq(self.current_freq)
            self.last_step_time = time.time()

        raw = self.tb.get_data()
        if raw is None:
            return

        # FFT → Power in dB
        win = np.hanning(FFT_SIZE)
        spec = np.fft.fftshift(np.fft.fft(raw*win, n=FFT_SIZE))
        psd = 20*np.log10(np.abs(spec)+1e-12)
        freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/self.sr)) + self.current_freq

        # update waterfall
        self.waterfall.append(psd)
        self.img.setImage(np.array(self.waterfall), autoLevels=False)

        # peak detection
        peaks, _ = find_peaks(psd, prominence=12)
        for p in peaks:
            f = freqs[p]
            band = classify_frequency(f)
            if band != "Unknown":
                entry = f"{f/1e6:.2f} MHz → {band} ({psd[p]:.1f} dB)"
                self.log_entries.append(entry)
                self.csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                          f/1e6, float(psd[p]), band])
                self.csv_file.flush()

        # update GUI log
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
