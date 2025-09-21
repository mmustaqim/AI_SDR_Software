#!/usr/bin/env python3
"""
gnuradio_auto_tune_gui.py

- Embeds a GNU Radio top_block (osmosdr source -> stream_to_vector -> vector_sink_c)
- Pulls blocks of IQ samples from vector_sink_c into Python
- Computes FFT (numpy) and displays realtime spectrum with pyqtgraph
- Automatically detects peaks and (optionally) re-centers HackRF to the strongest peak

Requirements:
  - gnuradio (with gr-osmosdr)
  - hackrf (device + drivers)
  - python3, pyqt5, pyqtgraph, numpy, scipy
"""

import sys
import time
import numpy as np
from scipy.signal import find_peaks

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# GNU Radio imports
from gnuradio import gr, blocks
import osmosdr

# ------------------ User-config ------------------
SAMPLE_RATE = 2.4e6        # HackRF sample rate (Hz) - adjust to your device capability
CENTER_FREQ = 915e6        # initial center (Hz) - change to band you want
FFT_SIZE = 2048            # FFT size (power of two)
VEC_PER_POLL = 1           # how many vectors to read when polling
POLL_INTERVAL_MS = 80      # GUI timer interval in ms
AUTO_TUNE_ENABLED = True   # enable automatic retuning
TUNE_PROMINENCE_DB = 6.0   # minimum prominence (dB) for a peak to be considered
TUNE_MIN_SNR_DB = 8.0      # minimal difference from median noise to allow retune
TUNE_HYSTERESIS_SEC = 1.0  # minimal time between auto-tunes
HACKRF_DEVICE_ARGS = "hackrf=0"  # osmosdr device args (choose index or serial)
# -------------------------------------------------

class GrTop(gr.top_block):
    def __init__(self, samp_rate, center_freq, fft_size):
        gr.top_block.__init__(self)

        # osmosdr source (HackRF via gr-osmosdr)
        self.src = osmosdr.source(args=HACKRF_DEVICE_ARGS)
        self.src.set_sample_rate(samp_rate)
        self.src.set_center_freq(center_freq)
        # optional gain control: start with moderate gain; the exact calls depend on driver
        try:
            self.src.set_gain_mode(False)
            self.src.set_gain(40)  # tune smaller/larger as needed
        except Exception:
            pass

        # Convert stream to vectors (vlen = fft_size)
        self.stream_to_vec = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size)
        # Vector sink to store vectors in memory (we'll poll from Python)
        self.vec_sink = blocks.vector_sink_c(fft_size)

        # Connect pipeline: src -> stream_to_vector -> vector_sink_c
        self.connect(self.src, self.stream_to_vec, self.vec_sink)

    def set_center_freq(self, freq_hz):
        """Set HackRF center frequency (safe wrapper)."""
        # osmosdr source exposes set_center_freq
        try:
            self.src.set_center_freq(freq_hz)
        except Exception as e:
            print("Failed to set center frequency:", e)

    def get_center_freq(self):
        # Not all osmosdr backends provide a getter; we store it manually by reading src settings if available
        try:
            return self.src.get_center_freq()
        except Exception:
            return None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, top_block: GrTop, sample_rate, fft_size):
        super().__init__()
        self.tb = top_block
        self.sr = sample_rate
        self.fft_size = fft_size
        self.center_freq = CENTER_FREQ
        self.last_tune_time = 0.0

        self.setWindowTitle("GNU Radio Auto-Tune Spectrum (HackRF)")
        self.resize(1000, 700)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout()
        central.setLayout(v)

        # pyqtgraph plot
        self.pg_plot = pg.PlotWidget(title="Realtime Spectrum (dBFS)")
        self.pg_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.pg_plot.setLabel('left', 'Power', units='dB')
        v.addWidget(self.pg_plot, 3)

        # control row
        ctrl = QtWidgets.QHBoxLayout()
        self.auto_cb = QtWidgets.QCheckBox("Auto-Tune")
        self.auto_cb.setChecked(AUTO_TUNE_ENABLED)
        ctrl.addWidget(self.auto_cb)

        self.freq_label = QtWidgets.QLabel(f"Center: {self.center_freq/1e6:.6f} MHz")
        ctrl.addWidget(self.freq_label)
        ctrl.addStretch()
        v.addLayout(ctrl)

        # plot curve
        self.curve = self.pg_plot.plot(pen='y')

        # Start a timer to poll vector_sink
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll_and_update)
        self.timer.start(POLL_INTERVAL_MS)

    def poll_and_update(self):
        # Get raw list from vector_sink
        try:
            data = self.tb.vec_sink.data()  # returns Python list of vectors (each vector is np.complex128 or complex)
        except Exception as e:
            print("Error reading vector sink:", e)
            return

        if len(data) == 0:
            return

        # Use the last vector available
        # vector_sink stores concatenated vectors; depending on GNURadio version you might get a 1D list
        # but vector_sink_c's data() returns a list-like; convert to numpy complex vector of length fft_size
        raw = np.array(data[-1], dtype=np.complex64)
        if raw.size != self.fft_size:
            # try to reshape or pick tail
            if raw.size > self.fft_size:
                raw = raw[-self.fft_size:]
            else:
                # skip if not full
                return

        # Apply window (to reduce spectral leakage)
        win = np.hanning(self.fft_size)
        vec = raw * win

        # FFT and power (dB)
        spec = np.fft.fftshift(np.fft.fft(vec, n=self.fft_size))
        psd = 20.0 * np.log10(np.abs(spec) + 1e-12)  # dB magnitude

        # Frequency axis (Hz)
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1.0 / self.sr)) + self.center_freq

        # Update plot
        self.curve.setData(freqs, psd)

        # Detect peaks (simple heuristic)
        try:
            # prominence in linear units: convert dB prominence to linear by using difference threshold in dB
            peaks, props = find_peaks(psd, prominence=TUNE_PROMINENCE_DB)
            if len(peaks) > 0:
                # choose the strongest peak
                peak_idx = peaks[np.argmax(psd[peaks])]
                peak_freq = freqs[peak_idx]
                peak_db = psd[peak_idx]

                # mark on plot (small red dot)
                # remove previous annotation by reusing a scatter plot item
                if not hasattr(self, 'peak_marker'):
                    self.peak_marker = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0))
                    self.pg_plot.addItem(self.peak_marker)
                self.peak_marker.setData([peak_freq], [peak_db])

                # Auto-tune logic: check SNR / prominence relative to median
                median_db = np.median(psd)
                if self.auto_cb.isChecked():
                    if (peak_db - median_db) >= TUNE_MIN_SNR_DB:
                        now = time.time()
                        if now - self.last_tune_time >= TUNE_HYSTERESIS_SEC:
                            # Retune center frequency to the detected peak (center on it)
                            new_center = float(peak_freq)
                            print(f"[AUTO-TUNE] Retuning to {new_center/1e6:.6f} MHz (peak {peak_db:.1f} dB, noise {median_db:.1f} dB)")
                            self.tb.set_center_freq(new_center)
                            self.center_freq = new_center
                            self.freq_label.setText(f"Center: {self.center_freq/1e6:.6f} MHz")
                            self.last_tune_time = now
        except Exception as e:
            print("Peak-detect exception:", e)

def main():
    # Create and start GNU Radio top block
    tb = GrTop(samp_rate=SAMPLE_RATE, center_freq=CENTER_FREQ, fft_size=FFT_SIZE)
    tb.start()  # starts background threads to feed vector_sink

    # Launch Qt app
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(tb, SAMPLE_RATE, FFT_SIZE)
    win.show()
    exit_code = app.exec_()

    # On exit, stop top block cleanly
    tb.stop()
    tb.wait()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
