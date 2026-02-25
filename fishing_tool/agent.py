from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import easyocr
import mss
import numpy as np
import pyautogui
import pygetwindow as gw
from paddleocr import PaddleOCR

from .config import FishingConfig


@dataclass
class TriggerResult:
    matched: bool
    keyword: Optional[str] = None
    action: Optional[str] = None
    button: Optional[str] = None
    text: str = ""


class FishingAgent:
    def __init__(self, config: FishingConfig):
        self.config = config
        self.last_trigger_time = 0.0
        self.rod_casted = False

        self.started_at = time.time()
        self.cast_timestamps: list[float] = []
        self.next_stats_print_at = self.started_at + max(1.0, self.config.stats_print_interval_sec)
        self.last_bite_seen_at = self.started_at
        self.last_no_bite_recover_at = 0.0
        self.last_nonempty_ocr_at = self.started_at
        self.last_ocr_empty_recover_at = 0.0

        if self.config.ocr_engine == "easyocr":
            self.reader = easyocr.Reader(config.languages, gpu=False)
            self.paddle_reader = None
        else:
            self.reader = None
            self.paddle_reader = PaddleOCR(
                use_angle_cls=False,
                lang=self.config.ocr_lang,
                show_log=False,
            )

    def _normalize(self, text: str) -> str:
        if self.config.case_sensitive:
            return text
        return text.lower()

    def _window_region(self) -> dict[str, int]:
        if self.config.region:
            return self.config.region
        if self.config.window_title_contains:
            windows = gw.getWindowsWithTitle(self.config.window_title_contains)
            if not windows:
                raise RuntimeError(f"Window not found: {self.config.window_title_contains}")
            w = windows[0]
            return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}
        with mss.mss() as sct:
            monitor = sct.monitors[1]
        return {
            "left": monitor["left"],
            "top": monitor["top"],
            "width": monitor["width"],
            "height": monitor["height"],
        }

    def _focus_region(self) -> dict[str, int]:
        base = self._window_region()
        ratio = self.config.focus_region_ratio
        if not ratio:
            return base

        x = int(base["left"] + base["width"] * float(ratio.get("x", 0.0)))
        y = int(base["top"] + base["height"] * float(ratio.get("y", 0.0)))
        w = int(base["width"] * float(ratio.get("width", 1.0)))
        h = int(base["height"] * float(ratio.get("height", 1.0)))

        return {
            "left": max(x, base["left"]),
            "top": max(y, base["top"]),
            "width": max(1, min(w, base["left"] + base["width"] - x)),
            "height": max(1, min(h, base["top"] + base["height"] - y)),
        }

    def _ocr_with_easyocr(self, img: np.ndarray) -> list[str]:
        if self.reader is None:
            return []
        return self.reader.readtext(img, detail=0, paragraph=True)

    def _ocr_with_paddle(self, img: np.ndarray) -> list[str]:
        if self.paddle_reader is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        result = self.paddle_reader.ocr(binary, cls=False)

        texts: list[str] = []
        if not result:
            return texts

        for line in result:
            if not line:
                continue
            for item in line:
                if not item or len(item) < 2:
                    continue
                text_info = item[1]
                if isinstance(text_info, (list, tuple)) and text_info:
                    texts.append(str(text_info[0]))
        return texts

    def capture_text(self) -> str:
        region = self._focus_region()
        with mss.mss() as sct:
            shot = sct.grab(region)
        img = np.array(shot)

        if self.config.ocr_engine == "easyocr":
            texts = self._ocr_with_easyocr(img)
        else:
            texts = self._ocr_with_paddle(img)

        return " ".join(texts).strip()

    def _select_button(self, normalized_text: str) -> str:
        for key, button in self.config.button_rules.items():
            k = key if self.config.case_sensitive else key.lower()
            if k in normalized_text:
                return "left" if button == "left" else "right"
        return "left" if self.config.default_button == "left" else "right"

    def _click(self, button: str) -> None:
        pyautogui.click(button=button)

    def _cast_once(self, button: str) -> None:
        self._click(button)
        self.rod_casted = True
        self.cast_timestamps.append(time.time())

    def _reel_once(self, button: str) -> None:
        self._click(button)
        self.rod_casted = False

    def _recast(self, button: str) -> None:
        self._reel_once(button)
        time.sleep(self.config.recast_delay_sec)
        self._cast_once(button)

    def _sync_state_from_text(self, normalized_text: str) -> None:
        cast_kw = self._normalize(self.config.cast_keyword)
        reel_kw = self._normalize(self.config.reel_keyword)
        if cast_kw and cast_kw in normalized_text:
            self.rod_casted = True
        if reel_kw and reel_kw in normalized_text:
            self.rod_casted = False

    def _resolve_action(self, matched_keyword: str) -> str:
        action = self.config.keyword_actions.get(matched_keyword)
        if not action:
            return "click"
        return action

    def _stats_snapshot(self) -> tuple[int, float, float]:
        cast_count = len(self.cast_timestamps)
        runtime = max(0.0, time.time() - self.started_at)
        if cast_count < 2:
            avg_cast_interval = 0.0
        else:
            intervals = [
                self.cast_timestamps[i] - self.cast_timestamps[i - 1]
                for i in range(1, cast_count)
            ]
            avg_cast_interval = sum(intervals) / len(intervals)
        return cast_count, avg_cast_interval, runtime

    def _emit_stats(self, final: bool = False) -> None:
        cast_count, avg_cast_interval, runtime = self._stats_snapshot()
        tag = "STATS-FINAL" if final else "STATS"
        msg = (
            f"[{tag}] casts={cast_count} avg_cast_interval_sec={avg_cast_interval:.2f} "
            f"runtime_sec={runtime:.2f}"
        )
        print(msg)

        if self.config.stats_log_file:
            log_path = Path(self.config.stats_log_file)
            if not log_path.is_absolute():
                log_path = Path.cwd() / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def _touch_bite_presence(self, normalized_text: str) -> None:
        for kw in self.config.bite_presence_keywords:
            target = kw if self.config.case_sensitive else kw.lower()
            if target and target in normalized_text:
                self.last_bite_seen_at = time.time()
                return

    def _run_recover_action(self, action: str, reason_tag: str) -> None:
        button = "left" if self.config.default_button == "left" else "right"

        if action == "recast":
            self._recast(button)
            recover_note = "recast"
        elif action == "smart_recover":
            probe_click_at = time.time()
            self._click(button)
            time.sleep(max(0.05, self.config.smart_recover_probe_wait_sec))

            probe_text = self.capture_text()
            if self.config.print_ocr_text:
                print(f"[OCR-PROBE] {probe_text}")
            if probe_text.strip():
                self.last_nonempty_ocr_at = time.time()

            probe_normalized = self._normalize(probe_text)
            self._sync_state_from_text(probe_normalized)
            self._touch_bite_presence(probe_normalized)

            cast_kw = self._normalize(self.config.cast_keyword)
            reel_kw = self._normalize(self.config.reel_keyword)
            cast_hit = cast_kw and cast_kw in probe_normalized
            reel_hit = reel_kw and reel_kw in probe_normalized

            if reel_hit:
                self._cast_once(button)
                recover_note = "smart_recover(retrieved_then_cast)"
            elif cast_hit:
                self.rod_casted = True
                self.cast_timestamps.append(probe_click_at)
                recover_note = "smart_recover(thrown_ok)"
            elif not self.rod_casted:
                self._cast_once(button)
                recover_note = "smart_recover(unknown_then_cast)"
            else:
                recover_note = "smart_recover(unknown_keep)"
        elif action == "cast_if_idle":
            if not self.rod_casted:
                self._cast_once(button)
                recover_note = "cast_if_idle(casted)"
            else:
                recover_note = "cast_if_idle(skip_already_casted)"
        else:
            self._click(button)
            recover_note = "click"

        print(f"[RECOVER] {reason_tag} action={recover_note}")

    def _handle_no_bite_timeout(self) -> None:
        timeout = self.config.no_bite_timeout_sec
        if timeout is None or timeout <= 0:
            return

        now = time.time()
        if now - self.last_bite_seen_at < timeout:
            return

        cooldown = max(0.0, self.config.no_bite_recover_cooldown_sec)
        if now - self.last_no_bite_recover_at < cooldown:
            return

        self._run_recover_action(self.config.no_bite_timeout_action, "no_bite_timeout")
        self.last_no_bite_recover_at = now
        self.last_bite_seen_at = now

    def _handle_ocr_empty_timeout(self) -> None:
        timeout = self.config.ocr_empty_timeout_sec
        if timeout is None or timeout <= 0:
            return

        now = time.time()
        if now - self.last_nonempty_ocr_at < timeout:
            return

        cooldown = max(0.0, self.config.ocr_empty_recover_cooldown_sec)
        if now - self.last_ocr_empty_recover_at < cooldown:
            return

        self._run_recover_action(self.config.ocr_empty_timeout_action, "ocr_empty_timeout")
        self.last_ocr_empty_recover_at = now
        self.last_nonempty_ocr_at = now

    def step(self) -> TriggerResult:
        text = self.capture_text()
        if self.config.print_ocr_text:
            print(f"[OCR] {text}")

        if text.strip():
            self.last_nonempty_ocr_at = time.time()

        normalized_text = self._normalize(text)
        self._sync_state_from_text(normalized_text)
        self._touch_bite_presence(normalized_text)

        for kw in self.config.keywords:
            target = kw if self.config.case_sensitive else kw.lower()
            if target not in normalized_text:
                continue

            now = time.time()
            if now - self.last_trigger_time < self.config.cooldown_sec:
                return TriggerResult(matched=False, text=text)

            action = self._resolve_action(kw)
            button = self._select_button(normalized_text)

            if action == "recast":
                self._recast(button)
            elif action == "cast_if_idle":
                if not self.rod_casted:
                    self._cast_once(button)
            elif action == "reel_only":
                self._reel_once(button)
            else:
                self._click(button)

            self.last_trigger_time = now
            self.last_bite_seen_at = now
            return TriggerResult(
                matched=True,
                keyword=kw,
                action=action,
                button=button,
                text=text,
            )

        return TriggerResult(matched=False, text=text)

    def run(self) -> None:
        print("FishingAgent started. Press Ctrl+C to stop.")

        start_button = "left" if self.config.default_button == "left" else "right"
        if not self.rod_casted:
            print("[BOOT] no cast state detected, casting once.")
            self._cast_once(start_button)
            time.sleep(self.config.recast_delay_sec)

        while True:
            try:
                result = self.step()
                now = time.time()
                if result.matched:
                    print(
                        f"[TRIGGER] keyword={result.keyword} action={result.action} button={result.button} casted={self.rod_casted}"
                    )

                self._handle_no_bite_timeout()
                self._handle_ocr_empty_timeout()

                if now >= self.next_stats_print_at:
                    self._emit_stats(final=False)
                    self.next_stats_print_at = now + max(1.0, self.config.stats_print_interval_sec)
                time.sleep(self.config.interval_sec)
            except KeyboardInterrupt:
                self._emit_stats(final=True)
                print("Stopped.")
                break
            except Exception as e:
                print(f"[WARN] {e}")
                time.sleep(max(self.config.interval_sec, 0.2))