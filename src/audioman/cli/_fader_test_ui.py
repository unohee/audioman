# Created: 2026-04-27
# Purpose: PyQt6 multitrack mixer UI for audioman fader-test.

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from audioman.core.multitrack_player import MultitrackPlayer


# Slider scale: -60dB ~ +12dB → 0 ~ 720 (0.1 dB resolution)
SLIDER_MIN_DB = -60.0
SLIDER_MAX_DB = 12.0
SLIDER_RESOLUTION = 10  # 0.1 dB

def db_to_slider(db: float) -> int:
    return int(round((db - SLIDER_MIN_DB) * SLIDER_RESOLUTION))

def slider_to_db(value: int) -> float:
    return SLIDER_MIN_DB + value / SLIDER_RESOLUTION


class TrackStrip(QWidget):
    """단일 트랙: 이름 라벨 + vertical fader + dB readout + M/S 버튼 + RMS meter."""

    def __init__(self, player: MultitrackPlayer, track_index: int, parent=None):
        super().__init__(parent)
        self.player = player
        self.track_index = track_index
        self.track = player.tracks[track_index]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(3)

        # 트랙 이름 (긴 이름 회전 또는 잘라쓰기)
        name = self.track.name
        if len(name) > 14:
            name = name[:13] + "…"
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        name_label.setWordWrap(True)
        name_label.setFixedHeight(28)
        layout.addWidget(name_label)

        # M/S buttons
        ms_row = QHBoxLayout()
        ms_row.setSpacing(2)
        self.mute_btn = QPushButton("M")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setFixedSize(22, 18)
        self.mute_btn.setStyleSheet(
            "QPushButton{font-size:9px;}"
            "QPushButton:checked{background:#c33;color:white;}"
        )
        self.mute_btn.toggled.connect(self._on_mute)

        self.solo_btn = QPushButton("S")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setFixedSize(22, 18)
        self.solo_btn.setStyleSheet(
            "QPushButton{font-size:9px;}"
            "QPushButton:checked{background:#cc3;color:black;}"
        )
        self.solo_btn.toggled.connect(self._on_solo)

        ms_row.addStretch()
        ms_row.addWidget(self.mute_btn)
        ms_row.addWidget(self.solo_btn)
        ms_row.addStretch()
        layout.addLayout(ms_row)

        # RMS meter (수평 진행바, 끝에서 위로 올라가는 모양 대신 옆으로)
        self.meter = QProgressBar()
        self.meter.setOrientation(Qt.Orientation.Vertical)
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setTextVisible(False)
        self.meter.setFixedWidth(8)
        self.meter.setMinimumHeight(180)
        self.meter.setStyleSheet(
            "QProgressBar{background:#222;border:1px solid #444;}"
            "QProgressBar::chunk{background:#3c3;}"
        )

        # Vertical fader
        self.fader = QSlider(Qt.Orientation.Vertical)
        self.fader.setMinimum(db_to_slider(SLIDER_MIN_DB))
        self.fader.setMaximum(db_to_slider(SLIDER_MAX_DB))
        self.fader.setValue(db_to_slider(0.0))
        self.fader.setMinimumHeight(180)
        self.fader.setFixedWidth(24)
        self.fader.valueChanged.connect(self._on_fader)
        # 우클릭으로 0dB 리셋
        self.fader.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.fader.customContextMenuRequested.connect(lambda *_: self.fader.setValue(db_to_slider(0.0)))

        fader_row = QHBoxLayout()
        fader_row.setSpacing(2)
        fader_row.addStretch()
        fader_row.addWidget(self.meter)
        fader_row.addWidget(self.fader)
        fader_row.addStretch()
        layout.addLayout(fader_row)

        # dB readout
        self.db_label = QLabel("0.0 dB")
        self.db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.db_label.setStyleSheet("font-size: 9px;")
        self.db_label.setFixedHeight(14)
        layout.addWidget(self.db_label)

        self.setFixedWidth(70)

    def _on_fader(self, value: int):
        db = slider_to_db(value)
        self.player.set_gain_db(self.track_index, db)
        self.db_label.setText(f"{db:+.1f} dB")

    def _on_mute(self, checked: bool):
        self.player.set_muted(self.track_index, checked)
        if checked and self.solo_btn.isChecked():
            self.solo_btn.setChecked(False)

    def _on_solo(self, checked: bool):
        self.player.set_soloed(self.track_index, checked)
        if checked and self.mute_btn.isChecked():
            self.mute_btn.setChecked(False)

    def reset(self):
        self.fader.setValue(db_to_slider(0.0))
        self.mute_btn.setChecked(False)
        self.solo_btn.setChecked(False)

    def update_meter(self):
        # 트랙 RMS를 0-100으로 변환 (dB 스케일, -60~0 → 0~100)
        rms = self.track.rms
        if rms <= 1e-6:
            self.meter.setValue(0)
            return
        db = 20.0 * math.log10(max(rms, 1e-6))
        # -60..0 → 0..100
        pct = max(0, min(100, int((db + 60) * 100 / 60)))
        self.meter.setValue(pct)
        # peak 빨간색 처리
        peak = self.track.peak
        if peak >= 1.0:
            self.meter.setStyleSheet(
                "QProgressBar{background:#222;border:1px solid #c33;}"
                "QProgressBar::chunk{background:#c33;}"
            )
        elif peak >= 0.9:
            self.meter.setStyleSheet(
                "QProgressBar{background:#222;border:1px solid #444;}"
                "QProgressBar::chunk{background:#cc3;}"
            )
        else:
            self.meter.setStyleSheet(
                "QProgressBar{background:#222;border:1px solid #444;}"
                "QProgressBar::chunk{background:#3c3;}"
            )

    def set_gain_external(self, db: float):
        """외부에서 (Load JSON 등) gain 변경. valueChanged signal로 player에 전파됨."""
        self.fader.setValue(db_to_slider(db))


class FaderTestWindow(QMainWindow):
    def __init__(self, player: MultitrackPlayer, source_dir: Path | None = None):
        super().__init__()
        self.player = player
        self.source_dir = source_dir
        self.setWindowTitle(f"audioman fader-test — {len(player.tracks)} tracks "
                            f"({player.duration_sec:.0f}s)")

        # Dark palette
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window, QColor(40, 40, 40))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        pal.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        pal.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        pal.setColor(QPalette.ColorRole.Button, QColor(60, 60, 60))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        self.setPalette(pal)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)

        # === Top: transport ===
        transport = QFrame()
        transport.setFrameShape(QFrame.Shape.StyledPanel)
        transport_l = QHBoxLayout(transport)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedWidth(80)
        self.play_btn.clicked.connect(self._toggle_play)
        transport_l.addWidget(self.play_btn)

        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setFixedWidth(70)
        self.stop_btn.clicked.connect(self._stop)
        transport_l.addWidget(self.stop_btn)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, int(player.duration_sec * 10))  # 0.1s resolution
        self.position_slider.setValue(0)
        self.position_slider.sliderReleased.connect(self._on_seek)
        self.position_slider.sliderPressed.connect(self._on_seek_start)
        self._seeking = False
        transport_l.addWidget(self.position_slider)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(110)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        transport_l.addWidget(self.time_label)

        # Master meter
        self.master_meter = QProgressBar()
        self.master_meter.setRange(0, 100)
        self.master_meter.setFixedWidth(120)
        self.master_meter.setTextVisible(False)
        self.master_meter.setStyleSheet(
            "QProgressBar{background:#111;border:1px solid #444;}"
            "QProgressBar::chunk{background:#3c3;}"
        )
        transport_l.addWidget(QLabel("Master:"))
        transport_l.addWidget(self.master_meter)

        self.master_peak_label = QLabel("0.0 dB")
        self.master_peak_label.setFixedWidth(80)
        self.master_peak_label.setStyleSheet("font-family: monospace;")
        transport_l.addWidget(self.master_peak_label)

        main_layout.addWidget(transport)

        # === Center: track strips (scrollable) ===
        strips_widget = QWidget()
        strips_layout = QHBoxLayout(strips_widget)
        strips_layout.setSpacing(1)
        strips_layout.setContentsMargins(4, 4, 4, 4)

        self.strips: list[TrackStrip] = []
        for i in range(len(player.tracks)):
            strip = TrackStrip(player, i)
            strips_layout.addWidget(strip)
            self.strips.append(strip)
        strips_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(strips_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll, stretch=1)

        # === Bottom: actions ===
        actions = QFrame()
        actions_l = QHBoxLayout(actions)

        self.save_btn = QPushButton("💾 Save Gains JSON")
        self.save_btn.clicked.connect(self._save_gains)
        actions_l.addWidget(self.save_btn)

        self.load_btn = QPushButton("📂 Load Gains JSON")
        self.load_btn.clicked.connect(self._load_gains)
        actions_l.addWidget(self.load_btn)

        self.reset_btn = QPushButton("⟳ Reset All to 0 dB")
        self.reset_btn.clicked.connect(self._reset_all)
        actions_l.addWidget(self.reset_btn)

        self.unmute_btn = QPushButton("🔊 Clear Mute/Solo")
        self.unmute_btn.clicked.connect(self._clear_mute_solo)
        actions_l.addWidget(self.unmute_btn)

        actions_l.addStretch()

        self.status_label = QLabel("ready")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        actions_l.addWidget(self.status_label)

        main_layout.addWidget(actions)

        # === Update timer ===
        self.timer = QTimer()
        self.timer.setInterval(50)  # 20 Hz
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self.resize(min(1500, 80 * len(player.tracks) + 100), 380)

    # ------------------------------------------------------------------
    # Transport handlers
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if self.player.is_playing:
            self.player.pause()
            self.play_btn.setText("▶ Play")
        else:
            self.player.play()
            self.play_btn.setText("⏸ Pause")

    def _stop(self):
        self.player.stop()
        self.play_btn.setText("▶ Play")
        self.position_slider.setValue(0)

    def _on_seek_start(self):
        self._seeking = True

    def _on_seek(self):
        self._seeking = False
        pos_sec = self.position_slider.value() / 10.0
        self.player.seek(pos_sec)

    def _tick(self):
        # 모든 strip의 meter 업데이트
        for s in self.strips:
            s.update_meter()
        # Master meter
        rms = self.player.master_rms
        if rms > 1e-6:
            db = 20.0 * math.log10(rms)
            pct = max(0, min(100, int((db + 60) * 100 / 60)))
            self.master_meter.setValue(pct)
        else:
            self.master_meter.setValue(0)
        # Peak label + clipping warning
        peak = self.player.master_peak
        if peak <= 1e-6:
            peak_db = -60.0
        else:
            peak_db = 20.0 * math.log10(peak)
        clipping = self.player.clipping_count > 0
        clip_str = f" CLIP×{self.player.clipping_count}" if clipping else ""
        self.master_peak_label.setText(f"{peak_db:+.1f} dB{clip_str}")
        if peak >= 1.0:
            self.master_peak_label.setStyleSheet("color: red; font-family: monospace; font-weight: bold;")
        elif peak >= 0.9:
            self.master_peak_label.setStyleSheet("color: yellow; font-family: monospace;")
        else:
            self.master_peak_label.setStyleSheet("color: #ddd; font-family: monospace;")
        # Position
        if not self._seeking and self.player.is_playing:
            pos = self.player.get_position_sec()
            self.position_slider.setValue(int(pos * 10))
        pos = self.player.get_position_sec()
        total = self.player.duration_sec
        self.time_label.setText(f"{_fmt_time(pos)} / {_fmt_time(total)}")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _save_gains(self):
        # 기본 저장 위치: source_dir 옆에 .audioman/fader_test/<timestamp>.json
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.source_dir:
            default_dir = self.source_dir.parent / ".audioman" / "fader_test"
            default_dir.mkdir(parents=True, exist_ok=True)
            default_path = str(default_dir / f"gains_{ts}.json")
        else:
            default_path = f"gains_{ts}.json"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save gains JSON", default_path, "JSON files (*.json)"
        )
        if not path:
            return

        data = {
            "version": 1,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": str(self.source_dir) if self.source_dir else None,
            "n_tracks": len(self.player.tracks),
            "duration_sec": self.player.duration_sec,
            "sample_rate": self.player.sample_rate,
            "master_gain_db": round(self.player.master_gain_db, 2),
            "gains": self.player.export_gains(),
            "state": self.player.export_state(),
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        self.status_label.setText(f"saved: {Path(path).name}")

    def _load_gains(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load gains JSON", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        gains = data.get("gains", data)
        if not isinstance(gains, dict):
            QMessageBox.warning(self, "Load failed", "JSON에 'gains' dict가 없습니다.")
            return
        # strip의 fader 값을 변경 (signal로 player에도 전파됨)
        matched = 0
        for s in self.strips:
            if s.track.name in gains:
                s.set_gain_external(float(gains[s.track.name]))
                matched += 1
        self.status_label.setText(f"loaded {matched} gains from {Path(path).name}")

    def _reset_all(self):
        for s in self.strips:
            s.reset()
        self.player.clipping_count = 0
        self.status_label.setText("reset all to 0 dB")

    def _clear_mute_solo(self):
        for s in self.strips:
            s.mute_btn.setChecked(False)
            s.solo_btn.setChecked(False)
        self.status_label.setText("cleared mute/solo")

    def closeEvent(self, event):
        self.timer.stop()
        self.player.stop()
        super().closeEvent(event)


def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"
