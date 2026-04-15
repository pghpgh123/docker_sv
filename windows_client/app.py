import os
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests
import sounddevice as sd
import soundfile as sf
import websocket
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


SAMPLE_RATE = 16000
CHANNELS = 1
DEFAULT_BLOCK_MS = 100
HIGH_ACCURACY_BLOCK_MS = 200
CONTROL_MIN_HEIGHT = 28
BUTTON_MIN_HEIGHT = 30
RULE_PAIR_GROUPS = 4
SEQ_COL_WIDTH = 46
BOTTOM_SAFE_PX = 28


@dataclass
class ServerEndpoints:
    base_http: str
    ws_url: str
    file_url: str
    learning_url: str


def build_endpoints(base_url: str) -> ServerEndpoints:
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Base URL must start with http:// or https://")

    host = parsed.hostname
    port = parsed.port
    if not host:
        raise ValueError("Invalid host")

    http_base = f"{parsed.scheme}://{host}:{port}" if port else f"{parsed.scheme}://{host}"
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_base = f"{ws_scheme}://{host}:{port}" if port else f"{ws_scheme}://{host}"

    return ServerEndpoints(
        base_http=http_base,
        ws_url=f"{ws_base}/ws/transcribe",
        file_url=f"{http_base}/transcribe/file",
        learning_url=f"{http_base}/learning/confirm",
    )


class WsWorker(QThread):
    sig_connected = Signal()
    sig_disconnected = Signal(str)
    sig_message = Signal(dict)
    sig_log = Signal(str)

    def __init__(self, ws_url: str) -> None:
        super().__init__()
        self.ws_url = ws_url
        self._ws_app = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        self._ws_app = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_app.run_forever(ping_interval=20, ping_timeout=10)

    def close(self) -> None:
        self._stop_event.set()
        if self._ws_app is not None:
            self._ws_app.close()

    def send_json(self, payload: dict) -> None:
        if self._ws_app is None:
            return
        try:
            self._ws_app.send(json.dumps(payload))
        except Exception as exc:
            self.sig_log.emit(f"send_json failed: {exc}")

    def send_binary(self, data: bytes) -> None:
        if self._ws_app is None:
            return
        try:
            self._ws_app.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception as exc:
            self.sig_log.emit(f"send_binary failed: {exc}")

    def _on_open(self, _ws) -> None:
        self.sig_connected.emit()
        self.sig_log.emit("WebSocket 已连接")

    def _on_message(self, _ws, message) -> None:
        try:
            payload = json.loads(message)
            self.sig_message.emit(payload)
        except Exception:
            self.sig_log.emit(f"raw message: {message}")

    def _on_error(self, _ws, error) -> None:
        self.sig_log.emit(f"WebSocket 错误: {error}")

    def _on_close(self, _ws, status_code, close_msg) -> None:
        msg = f"WebSocket 已关闭: code={status_code}, msg={close_msg}"
        self.sig_disconnected.emit(msg)


class MainWindow(QMainWindow):
    sig_rtsp_stream_message = Signal(dict)
    sig_rtsp_stream_log = Signal(str)
    sig_rtsp_stream_finished = Signal()

    CONFIG_PATH = os.path.expanduser("~/.sv_client_config.json")
    RTSP_STRATEGIES = [
        ("conservative", "保守RTSP"),
        ("adaptive", "自适应VAD（近麦克风）"),
        ("mirror_mic", "完全同麦克风"),
        ("final_whisper", "Whisper large-v3（仅最终文本）"),
    ]

    def _init_rtsp_stream_vars(self) -> None:
        self.rtsp_stream_ws = None
        self.rtsp_stream_thread = None
        self.rtsp_stream_running = False
        self.rtsp_seq_base = 0
        self.rtsp_display_current_seq: Optional[int] = None
        self.rtsp_display_current_text = ""
        self.rtsp_display_last_full_text = ""
        self.rtsp_display_history: list[str] = []
        self.rtsp_display_last_activity_ts = 0.0
        self.rtsp_pending_final_text = ""
        self.rtsp_pending_final_from_server = False
        self.rtsp_last_closed_seq: Optional[int] = None
        self.rtsp_closed_seq_queue: list[int] = []
        self.rtsp_idle_dot_phase = 0
        self.rtsp_idle_wait_logged = False

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SenseVoice 联调客户端")

        # 默认窗口为1600x900，适配125%缩放下的1920x1080屏幕
        self.setMinimumSize(1024, 600)
        self.setMaximumSize(1920, 1080)
        self.resize(1600, 900)

        self.ws_worker: Optional[WsWorker] = None
        self.endpoints: Optional[ServerEndpoints] = None
        self.current_session_id: str = ""
        self._last_partial_line: str = ""
        self._partial_history_by_seq: dict[int, list[str]] = {}
        self._partial_last_full_by_seq: dict[int, str] = {}
        self._partial_row_by_seq: dict[int, int] = {}
        self._final_lines_by_seq: dict[int, str] = {}
        self._partial_seq_order: list[int] = []
        self._final_seq_order: list[int] = []
        self._final_row_by_seq: dict[int, int] = {}
        self._confirmed_final_seq: set[int] = set()
        self._updating_final_table = False
        self._syncing_table_scroll = False

        self.input_stream = None
        self.mic_running = False
        self._restore_top_height = 380

        # 提前初始化 main_splitter，防止后续引用报错
        self.main_splitter = QSplitter(Qt.Vertical)
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self._build_ui()
        self._init_rtsp_stream_vars()
        self.sig_rtsp_stream_message.connect(self._on_rtsp_stream_message)
        self.sig_rtsp_stream_log.connect(self.log)
        self.sig_rtsp_stream_finished.connect(self._on_rtsp_stream_finished)
        self.rtsp_partial_timer = QTimer(self)
        self.rtsp_partial_timer.setInterval(350)
        self.rtsp_partial_timer.timeout.connect(self._on_rtsp_partial_tick)
        self.copy_shortcut = QShortcut(QKeySequence.Copy, self)
        self.copy_shortcut.activated.connect(self._copy_active_table_selection)
        self._apply_1080p_preset_layout()
        self._load_input_devices()
        self._load_user_config()

        self.base_url_edit.textChanged.connect(lambda _text: self._save_user_config())
        self.rtsp_url_edit.textChanged.connect(lambda _text: self._save_user_config())
        self.rtsp_strategy_combo.currentIndexChanged.connect(lambda _idx: self._save_user_config())
        self.rtsp_min_rms_spin.valueChanged.connect(lambda _value: self._save_user_config())
        self.wav_path_edit.textChanged.connect(lambda _text: self._save_user_config())

    def on_rtsp_stream_start(self) -> None:
        if self.rtsp_stream_running:
            return
        rtsp_url = self.rtsp_url_edit.text().strip()
        if not rtsp_url.startswith("rtsp://"):
            QMessageBox.warning(self, "无效地址", "请输入合法的RTSP地址")
            return
        if not self.endpoints:
            QMessageBox.warning(self, "未连接", "请先连接服务器")
            return
        self._save_user_config()
        current_max_seq = 0
        if self._partial_seq_order:
            current_max_seq = max(current_max_seq, max(self._partial_seq_order))
        if self._final_seq_order:
            current_max_seq = max(current_max_seq, max(self._final_seq_order))
        self.rtsp_seq_base = current_max_seq
        self.rtsp_display_current_seq = None
        self.rtsp_display_current_text = ""
        self.rtsp_display_last_full_text = ""
        self.rtsp_display_history = []
        self.rtsp_display_last_activity_ts = 0.0
        self.rtsp_pending_final_text = ""
        self.rtsp_pending_final_from_server = False
        self.rtsp_last_closed_seq = None
        self.rtsp_closed_seq_queue = []
        self.rtsp_idle_dot_phase = 0
        self.rtsp_idle_wait_logged = False
        final_only_mode = self._current_rtsp_strategy() == "final_whisper"
        if not final_only_mode:
            self._start_new_rtsp_display_sentence(initial=True)
            self.rtsp_partial_timer.start()
        ws_url = self.endpoints.ws_url.replace("/ws/transcribe", "/ws/rtsp_transcribe")
        self.rtsp_stream_running = True
        self.rtsp_stream_thread = threading.Thread(
            target=self._rtsp_stream_worker,
            args=(ws_url, rtsp_url, self._current_rtsp_strategy(), self.rtsp_min_rms_spin.value()),
            daemon=True,
        )
        self.rtsp_stream_thread.start()
        self.rtsp_stream_start_btn.setEnabled(False)
        self.rtsp_stream_stop_btn.setEnabled(True)
        self.log(
            f"[RTSP流式] 已启动，策略={self._current_rtsp_strategy_label()}，最小RMS={self.rtsp_min_rms_spin.value():.3f}"
        )

    def on_rtsp_stream_stop(self) -> None:
        self.rtsp_stream_running = False
        self.rtsp_partial_timer.stop()
        if self.rtsp_stream_ws:
            try:
                self.rtsp_stream_ws.close()
            except Exception:
                pass
        self.rtsp_stream_start_btn.setEnabled(True)
        self.rtsp_stream_stop_btn.setEnabled(False)
        self.log("[RTSP流式] 已停止")

    def _start_new_rtsp_display_sentence(self, initial: bool = False) -> None:
        next_seq = self.rtsp_seq_base + 1
        if self._partial_seq_order:
            next_seq = max(next_seq, max(self._partial_seq_order) + 1)
        if self._final_seq_order:
            next_seq = max(next_seq, max(self._final_seq_order) + 1)
        self.rtsp_display_current_seq = next_seq
        self.rtsp_display_current_text = ""
        self.rtsp_display_last_full_text = ""
        self.rtsp_display_history = []
        self.rtsp_display_last_activity_ts = time.monotonic()
        self.rtsp_pending_final_text = ""
        self.rtsp_pending_final_from_server = False
        self.rtsp_idle_dot_phase = 0
        self.rtsp_idle_wait_logged = False
        if next_seq not in self._partial_seq_order:
            self._partial_seq_order.append(next_seq)
        self._upsert_partial_row(next_seq, "")
        if initial:
            self.log(f"[RTSP流式] 初始化实时句子 seq={next_seq}")
        else:
            self.log(f"[RTSP流式] 开始下一条实时句子 seq={next_seq}")

    def _rtsp_stream_worker(self, ws_url: str, rtsp_url: str, rtsp_strategy: str, rtsp_min_rms: float) -> None:
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.settimeout(1.0)
            self.rtsp_stream_ws = ws
            ws.send(
                json.dumps(
                    {
                        "rtsp_url": rtsp_url,
                        "rtsp_strategy": rtsp_strategy,
                        "rtsp_min_rms": float(rtsp_min_rms),
                    }
                )
            )
            while self.rtsp_stream_running:
                try:
                    msg = ws.recv()
                    data = json.loads(msg)
                    error = str(data.get("error", "")).strip()
                    if error:
                        self.sig_rtsp_stream_log.emit(f"[RTSP流式] 服务端错误: {error}")
                        break
                    self.sig_rtsp_stream_message.emit(data)
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception as exc:
                    if self.rtsp_stream_running:
                        self.sig_rtsp_stream_log.emit(f"[RTSP流式] 异常: {exc}")
                    break
        except Exception as exc:
            self.sig_rtsp_stream_log.emit(f"[RTSP流式] 连接失败: {exc}")
        finally:
            self.rtsp_stream_running = False
            if self.rtsp_stream_ws:
                try:
                    self.rtsp_stream_ws.close()
                except Exception:
                    pass
            self.rtsp_stream_ws = None
            self.sig_rtsp_stream_finished.emit()

    def _on_rtsp_stream_message(self, payload: dict) -> None:
        mtype = payload.get("type")
        text = str(payload.get("text", "")).strip()
        if mtype in ("partial", "final") and text:
            self.log(f"[RTSP流式][{mtype}] seq={payload.get('seq')} text={text}")
        self._handle_asr_payload(payload, source_label="RTSP流式")

    def _copy_active_table_selection(self) -> None:
        widget = QApplication.focusWidget()
        while widget is not None and widget not in (self.partial_table, self.final_table):
            widget = widget.parentWidget()

        if widget not in (self.partial_table, self.final_table):
            return

        table = widget
        selected = table.selectedRanges()
        if not selected:
            return

        label = "实时文本：" if table == self.partial_table else "最终文本："
        lines = []
        seen_rows = set()
        for rng in selected:
            for row in range(rng.topRow(), rng.bottomRow() + 1):
                if row in seen_rows:
                    continue
                seen_rows.add(row)
                text_item = table.item(row, 1)
                text = text_item.text() if text_item else ""
                lines.append(f"{label}{text}")
        QApplication.clipboard().setText("\n".join(lines))
        self.log("已复制表格选中内容")

    def _on_rtsp_stream_finished(self) -> None:
        self.rtsp_partial_timer.stop()
        self.rtsp_stream_start_btn.setEnabled(True)
        self.rtsp_stream_stop_btn.setEnabled(False)

    def _commit_rtsp_final_row(self, seq_i: int, text: str) -> None:
        if seq_i not in self._final_seq_order:
            self._final_seq_order.append(seq_i)
        self._final_lines_by_seq[seq_i] = text
        self._upsert_final_row(seq_i, text, allow_confirm=False)

    def _handle_rtsp_server_final(self, text: str, seq: Optional[int] = None) -> None:
        final_text = str(text).strip()
        if self.rtsp_closed_seq_queue:
            target_seq = self.rtsp_closed_seq_queue.pop(0)
            self.log(f"[RTSP流式][final-apply] seq={target_seq} text={final_text or '<empty>'}")
            self._commit_rtsp_final_row(target_seq, final_text)
            if self.rtsp_last_closed_seq == target_seq:
                self.rtsp_last_closed_seq = None
            return

        if self.rtsp_display_current_seq is not None and self.rtsp_display_current_text.strip():
            self.rtsp_pending_final_text = final_text
            self.rtsp_pending_final_from_server = True
            self.log(
                f"[RTSP流式][final-buffered] seq={self.rtsp_display_current_seq} text={final_text or '<empty>'}"
            )
            return

        if seq is not None:
            seq_i = int(seq)
            self.log(f"[RTSP流式][final-direct] seq={seq_i} text={final_text or '<empty>'}")
            if final_text:
                self._commit_rtsp_final_row(seq_i, final_text)
            return

        self.log(f"[RTSP流式][final-drop] text={final_text or '<empty>'}")

    def _on_rtsp_partial_tick(self) -> None:
        if not self.rtsp_stream_running or self.rtsp_display_current_seq is None:
            return

        seq_i = self.rtsp_display_current_seq
        elapsed_ms = max(0.0, (time.monotonic() - self.rtsp_display_last_activity_ts) * 1000)
        if not self.rtsp_display_current_text:
            self.rtsp_idle_dot_phase = (self.rtsp_idle_dot_phase % 3) + 1
            animated = "." * self.rtsp_idle_dot_phase
            self._upsert_partial_row(seq_i, animated)
            if not self.rtsp_idle_wait_logged:
                self.log(f"[RTSP流式][waiting] seq={seq_i}")
                self.rtsp_idle_wait_logged = True
            return

        self.rtsp_idle_wait_logged = False
        self.rtsp_idle_dot_phase = (self.rtsp_idle_dot_phase % 3) + 1
        animated = f"{self.rtsp_display_current_text}{'.' * self.rtsp_idle_dot_phase}"
        self._upsert_partial_row(seq_i, animated)

        if elapsed_ms < self.vad_end_ms_spin.value():
            return

        if self.rtsp_pending_final_from_server:
            final_text = self.rtsp_pending_final_text.strip()
            should_wait_server_final = False
        else:
            final_text = self.rtsp_display_last_full_text.strip() or self.rtsp_display_current_text.strip()
            should_wait_server_final = True
        self.log(f"[RTSP流式][sentence-end] seq={seq_i} final={final_text or '<empty>'}")
        if not should_wait_server_final and final_text:
            self._commit_rtsp_final_row(seq_i, final_text)
        if should_wait_server_final and seq_i not in self.rtsp_closed_seq_queue:
            self.rtsp_closed_seq_queue.append(seq_i)
        self.rtsp_last_closed_seq = seq_i
        self.rtsp_pending_final_text = ""
        self.rtsp_pending_final_from_server = False
        self._start_new_rtsp_display_sentence(initial=False)

    def _save_user_config(self) -> None:
        cfg = {
            "base_url": self.base_url_edit.text().strip(),
            "rtsp_url": self.rtsp_url_edit.text().strip(),
            "rtsp_strategy": self._current_rtsp_strategy(),
            "rtsp_min_rms": float(self.rtsp_min_rms_spin.value()),
            "wav_path": self.wav_path_edit.text().strip(),
        }
        try:
            with open(self.CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_user_config(self) -> None:
        try:
            if os.path.exists(self.CONFIG_PATH):
                with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "base_url" in cfg:
                    self.base_url_edit.setText(cfg["base_url"])
                if "rtsp_url" in cfg:
                    self.rtsp_url_edit.setText(cfg["rtsp_url"])
                if "rtsp_strategy" in cfg:
                    self._set_rtsp_strategy(cfg["rtsp_strategy"])
                if "rtsp_min_rms" in cfg:
                    self.rtsp_min_rms_spin.setValue(float(cfg["rtsp_min_rms"]))
                if "wav_path" in cfg:
                    self.wav_path_edit.setText(cfg["wav_path"])
        except Exception:
            pass

    def _current_rtsp_strategy(self) -> str:
        value = self.rtsp_strategy_combo.currentData()
        return str(value or "conservative")

    def _current_rtsp_strategy_label(self) -> str:
        return self.rtsp_strategy_combo.currentText().strip() or "保守RTSP"

    def _set_rtsp_strategy(self, value: str) -> None:
        idx = self.rtsp_strategy_combo.findData(value)
        if idx >= 0:
            self.rtsp_strategy_combo.setCurrentIndex(idx)

    def _build_ui(self) -> None:
        root = QWidget()
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(5)

        root.setStyleSheet(
            "QGroupBox { font-size: 13px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 2px; }"
            "QLabel { font-size: 13px; }"
            "QPushButton { min-height: 30px; padding: 1px 8px; font-size: 13px; }"
            "QLineEdit { min-height: 28px; padding: 1px 6px; font-size: 13px; }"
            "QComboBox { min-height: 28px; padding: 1px 6px; font-size: 13px; }"
            "QSpinBox { min-height: 28px; padding: 1px 6px; font-size: 13px; }"
            "QCheckBox { min-height: 24px; font-size: 13px; }"
            "QPlainTextEdit { padding: 3px; font-size: 13px; }"
            "QTableWidget { font-size: 13px; gridline-color: #d0d0d0; }"
            "QHeaderView::section { padding: 3px 4px; min-height: 24px; font-size: 13px; }"
        )

        self.toggle_top_btn = QPushButton("隐藏上方控制区")

        conn_box = QGroupBox("服务连接")
        conn_layout = QGridLayout(conn_box)
        conn_layout.setContentsMargins(6, 6, 6, 6)
        conn_layout.setHorizontalSpacing(6)
        conn_layout.setVerticalSpacing(3)
        self.base_url_edit = QLineEdit("http://182.150.55.26:19000")
        self.connect_btn = QPushButton("连接")
        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.setEnabled(False)
        self.connection_status = QLabel("未连接")
        self.session_id_label = QLabel("-")
        conn_layout.addWidget(QLabel("服务地址"), 0, 0)
        conn_layout.addWidget(self.base_url_edit, 0, 1, 1, 3)
        conn_layout.addWidget(self.connect_btn, 1, 1)
        conn_layout.addWidget(self.disconnect_btn, 1, 2)
        conn_layout.addWidget(QLabel("状态"), 2, 0)
        conn_layout.addWidget(self.connection_status, 2, 1)
        conn_layout.addWidget(QLabel("会话ID"), 2, 2)
        conn_layout.addWidget(self.session_id_label, 2, 3)
        conn_layout.setColumnStretch(1, 2)
        conn_layout.setColumnStretch(3, 1)
        conn_box.setMinimumHeight(120)
        conn_box.setMaximumHeight(120)

        mic_box = QGroupBox("麦克风实时上传")
        mic_layout = QGridLayout(mic_box)
        mic_layout.setContentsMargins(6, 6, 6, 6)
        mic_layout.setHorizontalSpacing(5)
        mic_layout.setVerticalSpacing(2)
        self.device_combo = QComboBox()
        self.refresh_device_btn = QPushButton("刷新设备")
        self.start_mic_btn = QPushButton("开始")
        self.stop_mic_btn = QPushButton("停止")
        self.stop_mic_btn.setEnabled(False)
        self.mic_status = QLabel("空闲")
        self.high_accuracy_mode = QCheckBox("高精度")
        self.high_accuracy_mode.setChecked(True)
        self.vad_end_ms_spin = QSpinBox()
        self.vad_end_ms_spin.setRange(300, 3000)
        self.vad_end_ms_spin.setSingleStep(100)
        self.vad_end_ms_spin.setValue(2000)
        self.preset_low_latency_btn = QPushButton("1000")
        self.preset_balanced_btn = QPushButton("2000")
        self.preset_accuracy_btn = QPushButton("3000")
        mic_layout.addWidget(QLabel("设备"), 0, 0)
        mic_layout.addWidget(self.device_combo, 0, 1, 1, 2)
        mic_layout.addWidget(self.refresh_device_btn, 0, 3)
        mic_layout.addWidget(QLabel("静音ms"), 1, 0)
        mic_layout.addWidget(self.vad_end_ms_spin, 1, 1)
        mic_layout.addWidget(self.preset_low_latency_btn, 1, 2)
        mic_layout.addWidget(self.preset_balanced_btn, 1, 3)
        mic_layout.addWidget(self.preset_accuracy_btn, 1, 4)
        mic_layout.addWidget(self.start_mic_btn, 2, 1)
        mic_layout.addWidget(self.stop_mic_btn, 2, 2)
        mic_layout.addWidget(self.high_accuracy_mode, 2, 3)
        mic_layout.addWidget(self.mic_status, 2, 4)
        mic_layout.setColumnStretch(1, 2)
        mic_layout.setColumnStretch(4, 1)
        mic_box.setMinimumHeight(120)
        mic_box.setMaximumHeight(120)

        wav_box = QGroupBox("WAV 文件测试")
        wav_layout = QGridLayout(wav_box)
        wav_layout.setContentsMargins(6, 6, 6, 6)
        wav_layout.setHorizontalSpacing(5)
        wav_layout.setVerticalSpacing(2)
        self.wav_path_edit = QLineEdit()
        self.browse_wav_btn = QPushButton("选择")
        self.upload_wav_btn = QPushButton("上传")
        self.stream_wav_btn = QPushButton("回放")
        wav_layout.addWidget(self.wav_path_edit, 0, 0, 1, 3)
        wav_layout.addWidget(self.browse_wav_btn, 0, 3)
        wav_layout.addWidget(self.upload_wav_btn, 1, 1)
        wav_layout.addWidget(self.stream_wav_btn, 1, 2)
        wav_layout.setColumnStretch(0, 2)
        wav_layout.setColumnStretch(1, 1)
        wav_layout.setColumnStretch(2, 1)
        wav_layout.setColumnStretch(3, 1)
        wav_box.setMinimumHeight(120)
        wav_box.setMaximumHeight(120)

        rewrite_box = QGroupBox("领域纠错词典（在线生效）")
        rewrite_layout = QGridLayout(rewrite_box)
        rewrite_layout.setContentsMargins(6, 6, 6, 6)
        rewrite_layout.setHorizontalSpacing(6)
        rewrite_layout.setVerticalSpacing(4)
        self.rewrite_table = QTableWidget(0, RULE_PAIR_GROUPS * 2)
        self.rewrite_table.setHorizontalHeaderLabels(["原词", "替换词"] * RULE_PAIR_GROUPS)
        self.rewrite_table.verticalHeader().setVisible(False)
        self.rewrite_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.rewrite_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.rewrite_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        header = self.rewrite_table.horizontalHeader()
        for col in range(RULE_PAIR_GROUPS * 2):
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        self.rewrite_table.setMinimumHeight(150)

        self.add_rule_btn = QPushButton("新增词条")
        self.del_rule_btn = QPushButton("删除选中词条")
        self.import_rules_btn = QPushButton("导入词典")
        self.export_rules_btn = QPushButton("导出词典")
        self.load_rewrite_btn = QPushButton("读取词典")
        self.apply_rewrite_btn = QPushButton("应用词典")
        for btn in [
            self.add_rule_btn,
            self.del_rule_btn,
            self.import_rules_btn,
            self.export_rules_btn,
            self.load_rewrite_btn,
            self.apply_rewrite_btn,
        ]:
            btn.setFixedWidth(104)
            btn.setFixedHeight(24)
        self.stats_minutes_spin = QSpinBox()
        self.stats_minutes_spin.setRange(1, 120)
        self.stats_minutes_spin.setValue(10)
        self.query_session_stats_btn = QPushButton("查询当前会话统计")

        rewrite_header_row = QHBoxLayout()
        rewrite_header_row.setContentsMargins(0, 0, 0, 0)
        rewrite_header_row.setSpacing(6)
        rewrite_header_row.addWidget(QLabel("可双击编辑，支持增删，应用后实时生效"))
        rewrite_header_row.addWidget(self.add_rule_btn)
        rewrite_header_row.addWidget(self.del_rule_btn)
        rewrite_header_row.addWidget(self.import_rules_btn)
        rewrite_header_row.addWidget(self.export_rules_btn)
        rewrite_header_row.addWidget(self.load_rewrite_btn)
        rewrite_header_row.addWidget(self.apply_rewrite_btn)
        rewrite_header_row.addStretch(1)
        rewrite_layout.addLayout(rewrite_header_row, 0, 0, 1, 4)
        rewrite_layout.addWidget(self.rewrite_table, 1, 0, 1, 4)
        rewrite_box.setMaximumHeight(240)

        text_box = QGroupBox("识别文本")
        text_layout = QVBoxLayout(text_box)
        text_layout.setContentsMargins(4, 4, 4, 4)
        text_layout.setSpacing(3)
        self.partial_table = QTableWidget(0, 2)
        self.final_table = QTableWidget(0, 3)
        self.partial_table.setHorizontalHeaderLabels(["序号", "实时文本"])
        self.partial_table.verticalHeader().setVisible(False)
        self.partial_table.horizontalHeader().setVisible(False)
        self.partial_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.partial_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.partial_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.partial_table.setShowGrid(True)
        partial_header = self.partial_table.horizontalHeader()
        partial_header.setSectionResizeMode(0, QHeaderView.Fixed)
        partial_header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.partial_table.setColumnWidth(0, SEQ_COL_WIDTH)
        self.partial_table.verticalHeader().setDefaultSectionSize(32)

        self.final_table.setHorizontalHeaderLabels(["序号", "最终文本", "确认"])
        self.final_table.verticalHeader().setVisible(False)
        self.final_table.horizontalHeader().setVisible(False)
        self.final_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.final_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.final_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.final_table.setShowGrid(True)
        final_header = self.final_table.horizontalHeader()
        final_header.setSectionResizeMode(0, QHeaderView.Fixed)
        final_header.setSectionResizeMode(1, QHeaderView.Stretch)
        final_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.final_table.setColumnWidth(0, SEQ_COL_WIDTH)
        self.final_table.verticalHeader().setDefaultSectionSize(32)
        self.partial_table.setMinimumHeight(250)
        self.final_table.setMinimumHeight(250)

        self.partial_table.verticalScrollBar().valueChanged.connect(self.on_partial_scroll_changed)
        self.final_table.verticalScrollBar().valueChanged.connect(self.on_final_scroll_changed)
        self.clear_partial_btn = QPushButton("清空实时文本")
        self.clear_final_btn = QPushButton("清空最终文本")
        self.clear_partial_btn.setFixedWidth(102)
        self.clear_final_btn.setFixedWidth(102)
        self.clear_partial_btn.setFixedHeight(24)
        self.clear_final_btn.setFixedHeight(24)

        partial_header = QHBoxLayout()
        partial_header.setContentsMargins(0, 0, 0, 0)
        partial_header.setSpacing(4)
        partial_header.addWidget(QLabel("实时文本"))
        partial_header.addStretch(1)
        partial_header.addWidget(self.clear_partial_btn)

        final_header_row = QHBoxLayout()
        final_header_row.setContentsMargins(0, 0, 0, 0)
        final_header_row.setSpacing(4)
        final_header_row.addWidget(QLabel("最终文本"))
        final_header_row.addStretch(1)
        final_header_row.addWidget(self.clear_final_btn)

        partial_panel = QVBoxLayout()
        partial_panel.setContentsMargins(0, 0, 0, 0)
        partial_panel.setSpacing(2)
        partial_panel.addLayout(partial_header)
        partial_panel.addWidget(self.partial_table)

        final_panel = QVBoxLayout()
        final_panel.setContentsMargins(0, 0, 0, 0)
        final_panel.setSpacing(2)
        final_panel.addLayout(final_header_row)
        final_panel.addWidget(self.final_table)

        text_row = QHBoxLayout()
        text_row.setContentsMargins(0, 0, 0, 0)
        text_row.setSpacing(4)
        text_row.addLayout(partial_panel, 1)
        text_row.addLayout(final_panel, 1)
        text_layout.addLayout(text_row)

        log_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(4, 4, 4, 4)
        log_layout.setSpacing(3)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(250)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.setFixedHeight(24)
        self.log_stats_label = QLabel("统计窗口")

        log_top_row = QHBoxLayout()
        log_top_row.setContentsMargins(0, 0, 0, 0)
        log_top_row.setSpacing(4)
        log_top_row.addWidget(self.clear_log_btn)
        log_top_row.addStretch(1)
        log_top_row.addWidget(self.log_stats_label)
        log_top_row.addWidget(self.stats_minutes_spin)
        log_top_row.addWidget(self.query_session_stats_btn)

        log_layout.addLayout(log_top_row)
        log_layout.addWidget(self.log_text)

        controls = [
            self.base_url_edit,
            self.connect_btn,
            self.disconnect_btn,
            self.toggle_top_btn,
            self.device_combo,
            self.refresh_device_btn,
            self.start_mic_btn,
            self.stop_mic_btn,
            self.vad_end_ms_spin,
            self.preset_low_latency_btn,
            self.preset_balanced_btn,
            self.preset_accuracy_btn,
            self.wav_path_edit,
            self.browse_wav_btn,
            self.upload_wav_btn,
            self.stream_wav_btn,
            self.add_rule_btn,
            self.del_rule_btn,
            self.import_rules_btn,
            self.export_rules_btn,
            self.load_rewrite_btn,
            self.apply_rewrite_btn,
            self.stats_minutes_spin,
            self.query_session_stats_btn,
            self.clear_partial_btn,
            self.clear_final_btn,
            self.clear_log_btn,
        ]
        for widget in controls:
            if isinstance(widget, (QLineEdit, QComboBox, QSpinBox)):
                widget.setMinimumHeight(CONTROL_MIN_HEIGHT)
            else:
                widget.setMinimumHeight(BUTTON_MIN_HEIGHT)

        self.top_controls = QWidget()
        top_layout = QVBoxLayout(self.top_controls)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        # RTSP识别区
        rtsp_box = QGroupBox("RTSP音频识别")
        rtsp_layout = QGridLayout(rtsp_box)
        rtsp_layout.setContentsMargins(6, 6, 6, 6)
        rtsp_layout.setHorizontalSpacing(5)
        rtsp_layout.setVerticalSpacing(2)
        self.rtsp_url_edit = QLineEdit()
        self.rtsp_url_edit.setPlaceholderText("rtsp://...")
        self.rtsp_url_edit.setMinimumWidth(260)
        self.rtsp_strategy_combo = QComboBox()
        for value, label in self.RTSP_STRATEGIES:
            self.rtsp_strategy_combo.addItem(label, value)
        self.rtsp_strategy_combo.setMinimumHeight(CONTROL_MIN_HEIGHT)
        self.rtsp_min_rms_spin = QDoubleSpinBox()
        self.rtsp_min_rms_spin.setDecimals(3)
        self.rtsp_min_rms_spin.setRange(0.0, 0.200)
        self.rtsp_min_rms_spin.setSingleStep(0.002)
        self.rtsp_min_rms_spin.setValue(0.012)
        self.rtsp_min_rms_spin.setMinimumHeight(CONTROL_MIN_HEIGHT)
        # self.rtsp_recognize_btn = QPushButton("单次识别")
        self.rtsp_stream_start_btn = QPushButton("流式识别开始")
        self.rtsp_stream_stop_btn = QPushButton("停止")
        self.rtsp_stream_stop_btn.setEnabled(False)
        rtsp_layout.addWidget(QLabel("RTSP地址"), 0, 0)
        rtsp_layout.addWidget(self.rtsp_url_edit, 0, 1)
        rtsp_layout.addWidget(QLabel("策略"), 1, 0)
        rtsp_layout.addWidget(self.rtsp_strategy_combo, 1, 1)
        rtsp_layout.addWidget(QLabel("最小RMS"), 2, 0)
        rtsp_layout.addWidget(self.rtsp_min_rms_spin, 2, 1)
        # rtsp_layout.addWidget(self.rtsp_recognize_btn, 0, 2)
        rtsp_layout.addWidget(self.rtsp_stream_start_btn, 0, 3)
        rtsp_layout.addWidget(self.rtsp_stream_stop_btn, 0, 4)
        rtsp_layout.setColumnStretch(0, 0)
        rtsp_layout.setColumnStretch(1, 5)
        rtsp_layout.setColumnStretch(2, 0)
        rtsp_layout.setColumnStretch(3, 0)
        rtsp_layout.setColumnStretch(4, 0)
        rtsp_box.setMinimumHeight(116)
        rtsp_box.setMaximumHeight(116)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        # 连接区略宽，麦克风最窄，RTSP 明显放大
        top_row.addWidget(conn_box, 3)
        top_row.addWidget(mic_box, 1)   # 麦克风相对很窄
        top_row.addWidget(rtsp_box, 2)  # RTSP 区域约为麦克风 2 倍
        top_row.addWidget(wav_box, 3)
        top_layout.addLayout(top_row, 0)
        top_layout.addWidget(rewrite_box, 1)

        # 底部左右分割：识别文本 / 日志
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter.setHandleWidth(2)
        self.bottom_splitter.addWidget(text_box)
        self.bottom_splitter.addWidget(log_box)
        self.bottom_splitter.setStretchFactor(0, 3)
        self.bottom_splitter.setStretchFactor(1, 1)
        self.bottom_splitter.setSizes([1200, 400])

        # 上下分割：控制区 / 底部
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(self.top_controls)
        self.main_splitter.addWidget(self.bottom_splitter)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.addStretch(1)
        top_bar.addWidget(self.toggle_top_btn)

        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.main_splitter)
        self.setCentralWidget(root)

        # 信号绑定
        self.connect_btn.clicked.connect(self.on_connect)
        self.disconnect_btn.clicked.connect(self.on_disconnect)
        self.toggle_top_btn.clicked.connect(self.on_toggle_top_controls)
        self.refresh_device_btn.clicked.connect(self._load_input_devices)
        self.start_mic_btn.clicked.connect(self.on_start_mic)
        self.stop_mic_btn.clicked.connect(self.on_stop_mic)
        self.browse_wav_btn.clicked.connect(self.on_browse_wav)
        self.upload_wav_btn.clicked.connect(self.on_upload_wav_http)
        self.stream_wav_btn.clicked.connect(self.on_stream_wav_realtime)
        self.add_rule_btn.clicked.connect(self.on_add_rule)
        self.del_rule_btn.clicked.connect(self.on_delete_rule)
        self.import_rules_btn.clicked.connect(self.on_import_rules)
        self.export_rules_btn.clicked.connect(self.on_export_rules)
        self.load_rewrite_btn.clicked.connect(self.on_load_rewrite)
        self.apply_rewrite_btn.clicked.connect(self.on_apply_rewrite)
        self.query_session_stats_btn.clicked.connect(self.on_query_session_stats)
        self.preset_low_latency_btn.clicked.connect(lambda: self.on_set_preset(1000, "低延迟"))
        self.preset_balanced_btn.clicked.connect(lambda: self.on_set_preset(2000, "平衡"))
        self.preset_accuracy_btn.clicked.connect(lambda: self.on_set_preset(3000, "高精度"))
        self.clear_partial_btn.clicked.connect(self.on_clear_partial)
        self.clear_final_btn.clicked.connect(self.on_clear_final)
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        self.final_table.itemChanged.connect(self.on_final_table_item_changed)
        # self.rtsp_recognize_btn.clicked.connect(self.on_rtsp_recognize)
        self.rtsp_stream_start_btn.clicked.connect(self.on_rtsp_stream_start)
        self.rtsp_stream_stop_btn.clicked.connect(self.on_rtsp_stream_stop)

        self._set_rules_to_table([("寨上", "站上"), ("三脚痛", "三角筒"), ("新判断", "请判断")])



    def _apply_1080p_preset_layout(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            target_h = max(760, 920 - BOTTOM_SAFE_PX)
            self.resize(1600, target_h)
            top_h = int(target_h * 0.5)
            bottom_h = target_h - top_h
            self.main_splitter.setSizes([top_h, bottom_h])
            self.bottom_splitter.setSizes([int(1600 * 0.75), int(1600 * 0.25)])
            return

        g = screen.availableGeometry()
        target_w = min(g.width(), 1920)
        target_h = max(760, min(g.height(), 1080) - BOTTOM_SAFE_PX)
        self.resize(target_w, target_h)

        top_h = int(target_h * 0.38)
        self._restore_top_height = top_h
        bottom_h = max(260, target_h - top_h)
        self.main_splitter.setSizes([top_h, bottom_h])
        self.bottom_splitter.setSizes([int(target_w * 0.75), int(target_w * 0.25)])

    def on_toggle_top_controls(self) -> None:
        visible = self.top_controls.isVisible()
        self.top_controls.setVisible(not visible)
        if visible:
            self.toggle_top_btn.setText("显示上方控制区")
            self.main_splitter.setSizes([0, 1])
        else:
            self.toggle_top_btn.setText("隐藏上方控制区")
            top_h = min(self._restore_top_height, max(220, int(self.height() * 0.38)))
            self.main_splitter.setSizes([top_h, max(1, self.height() - top_h)])

    def _current_block_ms(self) -> int:
        return HIGH_ACCURACY_BLOCK_MS if self.high_accuracy_mode.isChecked() else DEFAULT_BLOCK_MS

    def on_set_preset(self, vad_end_ms: int, label: str) -> None:
        self.vad_end_ms_spin.setValue(vad_end_ms)
        self.log(f"已切换断句档位: {label}（{vad_end_ms}ms）")

    def _set_rules_to_table(self, rules: list[tuple[str, str]]) -> None:
        self.rewrite_table.setRowCount(0)
        for i in range(0, len(rules), RULE_PAIR_GROUPS):
            self._append_rule_row(rules[i : i + RULE_PAIR_GROUPS])
        self._refresh_rule_index()

    def _append_rule_row(self, pairs: Optional[list[tuple[str, str]]] = None) -> None:
        row = self.rewrite_table.rowCount()
        self.rewrite_table.insertRow(row)
        pairs = pairs or []

        for pair_idx in range(RULE_PAIR_GROUPS):
            col_src = pair_idx * 2
            col_dst = col_src + 1
            src, dst = ("", "")
            if pair_idx < len(pairs):
                src, dst = pairs[pair_idx]
            self.rewrite_table.setItem(row, col_src, QTableWidgetItem(src))
            self.rewrite_table.setItem(row, col_dst, QTableWidgetItem(dst))

    def _refresh_rule_index(self) -> None:
        return

    def _collect_rules_from_table(self) -> list[tuple[str, str]]:
        rules = []
        for row in range(self.rewrite_table.rowCount()):
            for pair_idx in range(RULE_PAIR_GROUPS):
                col_src = pair_idx * 2
                col_dst = col_src + 1
                src_item = self.rewrite_table.item(row, col_src)
                dst_item = self.rewrite_table.item(row, col_dst)
                src = (src_item.text().strip() if src_item else "")
                dst = (dst_item.text().strip() if dst_item else "")
                if src and dst:
                    rules.append((src, dst))
        return rules

    def on_add_rule(self) -> None:
        self._append_rule_row()
        self._refresh_rule_index()

    def on_delete_rule(self) -> None:
        row = self.rewrite_table.currentRow()
        col = self.rewrite_table.currentColumn()
        if row < 0 or col < 0:
            QMessageBox.warning(self, "未选中", "请先选中要删除的词条")
            return

        pair_col = (col // 2) * 2
        src_item = self.rewrite_table.item(row, pair_col)
        dst_item = self.rewrite_table.item(row, pair_col + 1)
        if src_item is not None:
            src_item.setText("")
        if dst_item is not None:
            dst_item.setText("")

        compact_rules = self._collect_rules_from_table()
        self._set_rules_to_table(compact_rules)
        self._refresh_rule_index()

    def on_import_rules(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入词典文件",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", f"读取文件失败: {exc}")
            return

        rules = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            for seg in [x.strip() for x in line.split(";") if x.strip()]:
                if "=>" not in seg:
                    continue
                src, dst = seg.split("=>", 1)
                src = src.strip()
                dst = dst.strip()
                if src and dst:
                    rules.append((src, dst))

        if not rules:
            QMessageBox.warning(self, "无有效规则", "文件中未解析到有效规则（格式: 原词=>替换词）")
            return

        self._set_rules_to_table(rules)
        self.log(f"已导入词典: {file_path}, 共 {len(rules)} 条")

    def on_export_rules(self) -> None:
        rules = self._collect_rules_from_table()
        if not rules:
            QMessageBox.warning(self, "词典为空", "当前没有可导出的有效规则")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出词典文件",
            "rewrite_rules.txt",
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for src, dst in rules:
                    f.write(f"{src}=>{dst}\n")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"写入文件失败: {exc}")
            return

        self.log(f"已导出词典: {file_path}, 共 {len(rules)} 条")

    def _start_payload(self) -> dict:
        # Keep min segment roughly aligned with silence threshold for better continuity.
        vad_end_ms = int(self.vad_end_ms_spin.value())
        # Avoid dropping short commands like "请查下" while still filtering tiny fragments.
        vad_min_segment_ms = max(400, int(vad_end_ms * 0.25))
        return {
            "type": "start",
            "sample_rate": SAMPLE_RATE,
            "vad_end_ms": vad_end_ms,
            "vad_min_segment_ms": vad_min_segment_ms,
        }

    def log(self, msg: str) -> None:
        # 仅在控件存在且未销毁时写日志，防止 RuntimeError
        if hasattr(self, "log_text") and self.log_text is not None:
            try:
                self.log_text.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")
            except RuntimeError:
                pass

    def _load_input_devices(self) -> None:
        self.device_combo.clear()
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                label = f"{idx}: {dev['name']}"
                self.device_combo.addItem(label, idx)
        self.log("输入设备已刷新")

    def on_connect(self) -> None:
        self._save_user_config()
        try:
            self.endpoints = build_endpoints(self.base_url_edit.text())
        except Exception as exc:
            QMessageBox.critical(self, "地址无效", str(exc))
            return

        self.ws_worker = WsWorker(self.endpoints.ws_url)
        self.ws_worker.sig_connected.connect(self._on_ws_connected)
        self.ws_worker.sig_disconnected.connect(self._on_ws_disconnected)
        self.ws_worker.sig_message.connect(self._on_ws_message)
        self.ws_worker.sig_log.connect(self.log)
        self.ws_worker.start()

    def on_disconnect(self) -> None:
        self.on_stop_mic()
        if self.ws_worker is not None:
            self.ws_worker.close()
            self.ws_worker.wait(1500)
            self.ws_worker = None
        self.connection_status.setText("未连接")
        self.current_session_id = ""
        self.session_id_label.setText("-")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.log("已断开")

    def _on_ws_connected(self) -> None:
        self.connection_status.setText("已连接")
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)

    def _on_ws_disconnected(self, msg: str) -> None:
        self.connection_status.setText("未连接")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.log(msg)

    def _on_ws_message(self, payload: dict) -> None:
        self._handle_asr_payload(payload, source_label="麦克风")

    def _handle_asr_payload(self, payload: dict, source_label: str) -> None:
        mtype = payload.get("type")
        if mtype == "partial":
            text = payload.get("text", "")
            seq = payload.get("seq")
            if seq is not None:
                seq_i = int(seq)
                if source_label == "RTSP流式":
                    if self.rtsp_display_current_seq is None:
                        self._start_new_rtsp_display_sentence(initial=True)
                    piece = str(text).strip()
                    prev_piece = self.rtsp_display_last_full_text
                    if piece and piece != prev_piece:
                        if not self.rtsp_display_history or self.rtsp_display_history[-1] != piece:
                            self.rtsp_display_history.append(piece)
                        self.rtsp_display_current_text = "...".join(self.rtsp_display_history)
                    self.rtsp_display_last_full_text = piece
                    if not self.rtsp_display_current_text:
                        self.rtsp_display_current_text = piece
                    self.rtsp_display_last_activity_ts = time.monotonic()
                    self.rtsp_idle_dot_phase = 0
                    self.rtsp_idle_wait_logged = False
                    if self.rtsp_display_current_text:
                        self._upsert_partial_row(self.rtsp_display_current_seq, self.rtsp_display_current_text)
                    return
                if seq_i not in self._partial_history_by_seq:
                    self._partial_seq_order.append(seq_i)
                    self._partial_history_by_seq[seq_i] = []
                    self._partial_last_full_by_seq[seq_i] = ""

                piece = str(text).strip()
                prev_piece = self._partial_last_full_by_seq.get(seq_i, "")
                history = self._partial_history_by_seq[seq_i]

                if piece and piece != prev_piece:
                    lcp_len = 0
                    max_len = min(len(prev_piece), len(piece))
                    while lcp_len < max_len and prev_piece[lcp_len] == piece[lcp_len]:
                        lcp_len += 1
                    delta_piece = piece[lcp_len:]
                    if delta_piece and (not history or history[-1] != delta_piece):
                        history.append(delta_piece)

                self._partial_last_full_by_seq[seq_i] = piece
                joined = "  ->  ".join(self._partial_history_by_seq.get(seq_i, []))
                self._upsert_partial_row(seq_i, joined)
            else:
                line = str(text)
                if line and line != self._last_partial_line:
                    self.log(f"partial(no-seq): {line}")
                    self._last_partial_line = line
        elif mtype == "final":
            text = payload.get("text", "")
            seq = payload.get("seq")
            if source_label == "RTSP流式":
                self._handle_rtsp_server_final(str(text), seq)
                return
            if text:
                if seq is not None:
                    seq_i = int(seq)
                    if seq_i not in self._final_lines_by_seq:
                        self._final_seq_order.append(seq_i)
                    self._final_lines_by_seq[seq_i] = text
                    self._upsert_final_row(seq_i, text)
                else:
                    self._append_local_final_row(text)
        elif mtype == "adaptive_mode":
            mode = payload.get("mode", "balanced")
            vad_end_ms = payload.get("vad_end_ms")
            min_seg = payload.get("vad_min_segment_ms")
            self.log(f"[{source_label}] 服务端自动模式: {mode}, 断句静音={vad_end_ms}ms, 最小语音段={min_seg}ms")
        elif mtype == "ready":
            self.current_session_id = payload.get("session_id", "")
            self.session_id_label.setText(self.current_session_id or "-")
            self.log(f"[{source_label}] {json.dumps(payload, ensure_ascii=False)}")
        elif mtype in ("started", "stopped", "warning", "error"):
            self.log(f"[{source_label}] {json.dumps(payload, ensure_ascii=False)}")
        else:
            self.log(f"[{source_label}] message: {payload}")

    def _ws_send_json(self, payload: dict) -> bool:
        if self.ws_worker is None:
            self.log("WebSocket 未连接")
            return False
        self.ws_worker.send_json(payload)
        return True

    def on_clear_partial(self) -> None:
        self._partial_history_by_seq.clear()
        self._partial_last_full_by_seq.clear()
        self._partial_row_by_seq.clear()
        self._partial_seq_order.clear()
        self._last_partial_line = ""
        self.partial_table.setRowCount(0)

    def _upsert_partial_row(self, seq_i: int, text: str) -> None:
        if seq_i in self._partial_row_by_seq:
            row = self._partial_row_by_seq[seq_i]
        else:
            row = self.partial_table.rowCount()
            self.partial_table.insertRow(row)
            self._partial_row_by_seq[seq_i] = row

        seq_item = QTableWidgetItem(str(seq_i))
        seq_item.setFlags(seq_item.flags() & ~Qt.ItemIsEditable)
        text_item = QTableWidgetItem(text)
        text_item.setFlags(text_item.flags() & ~Qt.ItemIsEditable)

        self.partial_table.setItem(row, 0, seq_item)
        self.partial_table.setItem(row, 1, text_item)
        self._scroll_tables_to_bottom()

    def on_clear_final(self) -> None:
        self._final_lines_by_seq.clear()
        self._final_seq_order.clear()
        self._final_row_by_seq.clear()
        self._confirmed_final_seq.clear()
        self._updating_final_table = True
        self.final_table.setRowCount(0)
        self._updating_final_table = False

    def on_partial_scroll_changed(self, value: int) -> None:
        if self._syncing_table_scroll:
            return
        self._syncing_table_scroll = True
        try:
            self.final_table.verticalScrollBar().setValue(value)
        finally:
            self._syncing_table_scroll = False

    def on_final_scroll_changed(self, value: int) -> None:
        if self._syncing_table_scroll:
            return
        self._syncing_table_scroll = True
        try:
            self.partial_table.verticalScrollBar().setValue(value)
        finally:
            self._syncing_table_scroll = False

    def _append_local_final_row(self, text: str) -> None:
        seq_i = max(self._final_seq_order) + 1 if self._final_seq_order else 1
        while seq_i in self._final_row_by_seq:
            seq_i += 1
        self._final_seq_order.append(seq_i)
        self._final_lines_by_seq[seq_i] = text
        self._upsert_final_row(seq_i, text, allow_confirm=False)

    def _upsert_final_row(self, seq_i: int, text: str, allow_confirm: bool = True) -> None:
        if seq_i in self._final_row_by_seq:
            row = self._final_row_by_seq[seq_i]
        else:
            row = self.final_table.rowCount()
            self.final_table.insertRow(row)
            self._final_row_by_seq[seq_i] = row

        self._updating_final_table = True

        seq_item = QTableWidgetItem(str(seq_i))
        seq_item.setFlags(seq_item.flags() & ~Qt.ItemIsEditable)
        text_item = QTableWidgetItem(text)
        text_item.setFlags(text_item.flags() & ~Qt.ItemIsEditable)

        check_item = self.final_table.item(row, 2)
        check_item_is_new = False
        if check_item is None:
            check_item = QTableWidgetItem()
            check_item_is_new = True
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        if not allow_confirm:
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        check_item.setFlags(flags)
        if check_item.checkState() != Qt.Checked:
            check_item.setCheckState(Qt.Unchecked)

        self.final_table.setItem(row, 0, seq_item)
        self.final_table.setItem(row, 1, text_item)
        if check_item_is_new:
            self.final_table.setItem(row, 2, check_item)

        self._updating_final_table = False
        self._scroll_tables_to_bottom()

    def _scroll_tables_to_bottom(self) -> None:
        self._syncing_table_scroll = True
        try:
            pbar = self.partial_table.verticalScrollBar()
            fbar = self.final_table.verticalScrollBar()
            pbar.setValue(pbar.maximum())
            fbar.setValue(fbar.maximum())
        finally:
            self._syncing_table_scroll = False

    def on_final_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_final_table:
            return
        if item.column() != 2:
            return
        if item.checkState() != Qt.Checked:
            return

        row = item.row()
        seq_item = self.final_table.item(row, 0)
        text_item = self.final_table.item(row, 1)
        if seq_item is None or text_item is None:
            return
        seq_text = seq_item.text().strip()
        if not seq_text.isdigit():
            return

        seq_i = int(seq_text)
        if seq_i in self._confirmed_final_seq:
            return

        final_text = text_item.text().strip()
        partial_text = self._partial_last_full_by_seq.get(seq_i, "")
        if not partial_text:
            partial_text = "".join(self._partial_history_by_seq.get(seq_i, []))

        self._confirmed_final_seq.add(seq_i)
        threading.Thread(
            target=self._submit_learning_confirm,
            args=(seq_i, partial_text, final_text),
            daemon=True,
        ).start()

    def _submit_learning_confirm(self, seq_i: int, partial_text: str, final_text: str) -> None:
        endpoints = self.endpoints
        if endpoints is None:
            self.log("学习上报失败: 未连接服务器")
            return
        try:
            payload = {
                "session_id": self.current_session_id,
                "seq": int(seq_i),
                "partial_text": partial_text,
                "final_text": final_text,
                "accepted": True,
            }
            resp = requests.post(endpoints.learning_url, json=payload, timeout=20)
            data = resp.json()
            if resp.status_code != 200 or not data.get("ok"):
                self.log(f"学习上报失败: {data}")
                return

            added = data.get("added", [])
            extracted = data.get("extracted", [])
            if added:
                pairs = ", ".join([f"{x.get('src')}=>{x.get('dst')}" for x in added])
                self.log(f"学习成功(seq={seq_i})，新增纠错词: {pairs}")
                self.on_load_rewrite()
            else:
                self.log(f"学习确认已记录(seq={seq_i})，本次未达到入库阈值")
            if extracted:
                pairs = ", ".join([f"{x.get('src')}=>{x.get('dst')}" for x in extracted])
                self.log(f"候选差异(seq={seq_i}): {pairs}")
        except Exception as exc:
            self.log(f"学习上报异常(seq={seq_i}): {exc}")

    def on_start_mic(self) -> None:
        if self.ws_worker is None:
            QMessageBox.warning(self, "未连接", "请先连接服务器")
            return
        if self.mic_running:
            return

        device_idx = self.device_combo.currentData()
        if device_idx is None:
            QMessageBox.warning(self, "无设备", "未检测到可用麦克风设备")
            return

        self._ws_send_json(self._start_payload())

        block_ms = self._current_block_ms()
        block_size = int(SAMPLE_RATE * block_ms / 1000)

        def audio_callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                self.log(f"麦克风状态: {status}")
            if self.ws_worker is not None and self.mic_running:
                self.ws_worker.send_binary(indata.tobytes())

        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=block_size,
            device=device_idx,
            callback=audio_callback,
        )
        self.input_stream.start()

        self.mic_running = True
        self.mic_status.setText("麦克风上传中")
        self.start_mic_btn.setEnabled(False)
        self.stop_mic_btn.setEnabled(True)
        self.log(f"麦克风实时上传已开始（分片 {block_ms}ms）")

    def on_stop_mic(self) -> None:
        if not self.mic_running:
            return

        try:
            if self.input_stream is not None:
                self.input_stream.stop()
                self.input_stream.close()
        finally:
            self.input_stream = None
            self.mic_running = False

        self._ws_send_json({"type": "stop"})
        self.mic_status.setText("麦克风空闲")
        self.start_mic_btn.setEnabled(True)
        self.stop_mic_btn.setEnabled(False)
        self.log("麦克风实时上传已停止")

    def on_browse_wav(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 WAV 文件", "", "WAV Files (*.wav)")
        if file_path:
            self.wav_path_edit.setText(file_path)
            self._save_user_config()

    def on_upload_wav_http(self) -> None:
        if self.endpoints is None:
            QMessageBox.warning(self, "未连接", "请先连接服务器")
            return

        wav_path = self.wav_path_edit.text().strip()
        if not wav_path:
            QMessageBox.warning(self, "未选择文件", "请先选择 WAV 文件")
            return

        def worker() -> None:
            try:
                endpoints = self.endpoints
                if endpoints is None:
                    self.log("未连接服务器")
                    return

                with open(wav_path, "rb") as f:
                    resp = requests.post(
                        endpoints.file_url,
                        files={"file": (wav_path.split("/")[-1], f, "audio/wav")},
                        timeout=120,
                    )
                data = resp.json()
                if resp.status_code != 200:
                    self.log(f"上传失败: {data}")
                    return
                text = str(data.get("text", "")).strip()
                if text:
                    self._append_local_final_row(text)
                self.log("WAV 上传成功")
            except Exception as exc:
                self.log(f"上传异常: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def closeEvent(self, event) -> None:
        self.on_rtsp_stream_stop()
        self.on_disconnect()
        super().closeEvent(event)

    def on_load_rewrite(self) -> None:
        if self.endpoints is None:
            QMessageBox.warning(self, "未连接", "请先连接服务器")
            return

        def worker() -> None:
            try:
                endpoints = self.endpoints
                if endpoints is None:
                    return
                resp = requests.get(f"{endpoints.base_http}/config/rewrite", timeout=15)
                data = resp.json()
                if resp.status_code != 200 or not data.get("ok"):
                    self.log(f"读取词典失败: {data}")
                    return
                rules = data.get("rules", [])
                parsed = []
                for item in rules:
                    src = str(item.get("src", "")).strip()
                    dst = str(item.get("dst", "")).strip()
                    if src and dst:
                        parsed.append((src, dst))
                if parsed:
                    self._set_rules_to_table(parsed)
                self.log("已读取服务端词典")
            except Exception as exc:
                self.log(f"读取词典异常: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def on_apply_rewrite(self) -> None:
        if self.endpoints is None:
            QMessageBox.warning(self, "未连接", "请先连接服务器")
            return

        rules = self._collect_rules_from_table()
        if not rules:
            QMessageBox.warning(self, "词典为空", "请至少保留一条有效词典规则")
            return
        text_rewrite = ";".join([f"{src}=>{dst}" for src, dst in rules])

        def worker() -> None:
            try:
                endpoints = self.endpoints
                if endpoints is None:
                    return
                resp = requests.post(
                    f"{endpoints.base_http}/config/rewrite",
                    json={"text_rewrite": text_rewrite},
                    timeout=15,
                )
                data = resp.json()
                if resp.status_code != 200 or not data.get("ok"):
                    self.log(f"应用词典失败: {data}")
                    return
                self.log("词典已应用，无需重启服务")
            except Exception as exc:
                self.log(f"应用词典异常: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def on_query_session_stats(self) -> None:
        if self.endpoints is None:
            QMessageBox.warning(self, "未连接", "请先连接服务器")
            return
        if not self.current_session_id:
            QMessageBox.warning(self, "无会话", "当前没有可查询的会话ID，请先连接并开始一次识别")
            return

        minutes = int(self.stats_minutes_spin.value())

        def worker() -> None:
            try:
                endpoints = self.endpoints
                if endpoints is None:
                    return
                url = f"{endpoints.base_http}/stats/adaptive/session/{self.current_session_id}?minutes={minutes}"
                resp = requests.get(url, timeout=15)
                data = resp.json()
                if resp.status_code != 200 or not data.get("ok"):
                    self.log(f"查询会话统计失败: {data}")
                    return
                self.log(f"会话统计: {json.dumps(data, ensure_ascii=False)}")
            except Exception as exc:
                self.log(f"查询会话统计异常: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def on_stream_wav_realtime(self) -> None:
        if self.ws_worker is None:
            QMessageBox.warning(self, "未连接", "请先连接服务器")
            return

        wav_path = self.wav_path_edit.text().strip()
        if not wav_path:
            QMessageBox.warning(self, "未选择文件", "请先选择 WAV 文件")
            return

        def worker() -> None:
            try:
                ws_worker = self.ws_worker
                if ws_worker is None:
                    self.log("WebSocket 未连接")
                    return

                audio, sr = sf.read(wav_path, dtype="int16", always_2d=False)
                if sr != SAMPLE_RATE:
                    self.log("WAV 实时回放要求 16kHz 音频")
                    return
                if getattr(audio, "ndim", 1) > 1:
                    audio = audio[:, 0]

                block_ms = self._current_block_ms()
                chunk_samples = int(SAMPLE_RATE * block_ms / 1000)

                self._ws_send_json(self._start_payload())
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i : i + chunk_samples]
                    if len(chunk) == 0:
                        break
                    ws_worker.send_binary(chunk.astype("int16").tobytes())
                    time.sleep(block_ms / 1000.0)

                self._ws_send_json({"type": "stop"})
                self.log(f"WAV 实时回放完成（分片 {block_ms}ms）")
            except Exception as exc:
                self.log(f"WAV 实时回放异常: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def closeEvent(self, event):  # noqa: N802
        self.on_disconnect()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
