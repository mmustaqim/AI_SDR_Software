#!/usr/bin/env python3
"""
GNU Radio HackRF Auto-Tune Spectrum GUI with Mini Classifier
"""

import sys, time
import numpy as np
from scipy.signal import find_peaks
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# GNU Radio
from gnuradio import gr, blocks
import osmosdr

# ------------------ Config ------------------
SAMPLE_RATE = 2e6
CENTER_FREQ = 915e6
FFT_SIZE = 2048
POLL_INTERVAL_MS = 100
AUTO_TUNE_ENABLED = True
TUNE_PROMINENCE_DB = 6.0
TUNE_MIN_SNR_DB = 8.0
TUNE_HYSTERESIS_SEC = 2.0
HACKRF_DEVICE_ARGS = "hackrf=0"
# --------------------------------------------

class GrTop(gr.top_block):
    def __init__(self, samp_rate, center_freq, fft_size):
        gr.top_block.__init__(self)
        self.src = osmosdr.source(args=HACKRF_DEVICE_ARGS)
        self.src.set_sample_rate(samp_rate)
        self.src.set_center_freq(center_freq)
        self.src.set_gain(40)

        self.stream_to_vec = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size)
        self.vec_sink = blocks.vector_sink_c(fft_size)

        self.connect(self.src, self.stream_to_vec, self.vec_sink)

    def set_center_freq(self, f):
        try: self.src.set_center_freq(f)
        except: pass

    def get_center_freq(self):
        try: return self.src.get_center_freq()
        except: return None

class SignalClassifier:
    """Very simple rule-based classifier (placeholder for ML model)."""
    def classify(self, freqs, psd_db):
        bw = freqs[-1] - freqs[0]
        noise_floor = np.median(psd_db)
        peaks, props = find_peaks(psd_db, prominence=6)
        if len(peaks) == 0:
            return "Noise"
        if len(peaks) > 5:
            return "Multi-carrier (GSM-like)"
        widths = []
        for p in peaks:
            half_max = psd_db[p] - 3
            left = np.where(psd_db[:p] < half_max)[0]
            right = np.where(psd_db[p:] < half_max)[0]
            w = (right[0] if len(right) else 1) + (p - (left[-1] if len(left) else p-1))
            widths.append(w)
        avg_width = np.mean(widths) if widths else 1
        frac = avg_width / len(psd_db)
        if frac > 0.05:
            return "Wideband (Wi-Fi/OFDM-like)"
        return "Narrowband (FM-like)"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, tb, samp_rate, fft_size):
        super().__init__()
        self.tb = tb
        self.sr = samp_rate
        self.fft_size = fft_size
        self.center_freq = CENTER_FREQ
        self.last_tune_time = 0
        self.classifier = SignalClassifier()

        self.setWindowTitle("HackRF Auto-Tune with Classifier")
        self.resize(1000, 700)
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(); cw.setLayout(v)

        self.plot = pg.PlotWidget(title="Realtime Spectrum")
        self.curve = self.plot.plot(pen='y')
        v.addWidget(self.plot, 3)

        ctrl = QtWidgets.QHBoxLayout()
        self.auto_cb = QtWidgets.QCheckBox("Auto-Tune"); self.auto_cb.setChecked(AUTO_TUNE_ENABLED)
        ctrl.addWidget(self.auto_cb)
        self.freq_label = QtWidgets.QLabel(f"Center: {self.center_freq/1e6:.3f} MHz")
        self.cls_label = QtWidgets.QLabel("Class: ---")
        ctrl.addWidget(self.freq_label); ctrl.addWidget(self.cls_label)
        ctrl.addStretch(); v.addLayout(ctrl)

        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.poll_and_update)
        self.timer.start(POLL_INTERVAL_MS)

    def poll_and_update(self):
        data = self.tb.vec_sink.data()
        if len(data) < self.fft_size: return
        raw = np.array(data[-self.fft_size:], dtype=np.complex64)
        win = np.hanning(self.fft_size)
        spec = np.fft.fftshift(np.fft.fft(raw*win, n=self.fft_size))
        psd = 20*np.log10(np.abs(spec)+1e-12)
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sr)) + self.center_freq

        self.curve.setData(freqs, psd)

        # classify
        label = self.classifier.classify(freqs, psd)
        self.cls_label.setText(f"Class: {label}")

        # peaks
        peaks, _ = find_peaks(psd, prominence=TUNE_PROMINENCE_DB)
        if len(peaks)==0: return
        idx = peaks[np.argmax(psd[peaks])]
        peak_freq, peak_db = freqs[idx], psd[idx]

        if self.auto_cb.isChecked():
            med = np.median(psd)
            if peak_db-med > TUNE_MIN_SNR_DB and time.time()-self.last_tune_time > TUNE_HYSTERESIS_SEC:
                self.tb.set_center_freq(float(peak_freq))
                self.center_freq = peak_freq
                self.freq_label.setText(f"Center: {self.center_freq/1e6:.3f} MHz")
                self.last_tune_time = time.time()

def main():
    tb = GrTop(SAMPLE_RATE, CENTER_FREQ, FFT_SIZE)
    tb.start()
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(tb, SAMPLE_RATE, FFT_SIZE); win.show()
    rc = app.exec_()
    tb.stop(); tb.wait()
    sys.exit(rc)

if __name__ == "__main__":
    main()
