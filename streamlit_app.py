"""
Qwen3 ASR 字幕生成器 - Streamlit Web 前端
Glass Morphism Dark UI | chatllm + Vulkan 推理後端

啟動：python -m streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st

# ── Page config（必須是第一個 Streamlit 呼叫）─────────────
st.set_page_config(
    page_title="Qwen3 ASR",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 路徑 ──────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
SETTINGS_FILE = BASE_DIR / "settings.json"
SRT_DIR       = BASE_DIR / "subtitles"
SRT_DIR.mkdir(exist_ok=True)

MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/dseditor/Collection/resolve/main/qwen3-asr-1.7b.bin"
)

SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]
SAMPLE_RATE = 16000


# ══════════════════════════════════════════════════════════
# Glass Morphism CSS
# ══════════════════════════════════════════════════════════

GLASS_CSS = """
<style>
/* ── 全域背景 ── */
.stApp {
    background: linear-gradient(135deg, #060610 0%, #0a0e1f 55%, #060c18 100%);
    font-family: 'Segoe UI', -apple-system, sans-serif;
}
header[data-testid="stHeader"] { background: transparent !important; }
.stMainBlockContainer { padding-top: 1rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(8, 12, 28, 0.85) !important;
    backdrop-filter: blur(24px);
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 5px 6px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 9px 28px;
    color: rgba(180,200,255,0.55);
    font-weight: 500;
    font-size: 0.95rem;
    transition: all 0.2s;
    border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,
        rgba(14,165,233,0.18), rgba(99,102,241,0.18)) !important;
    color: #93c5fd !important;
    border: 1px solid rgba(125,211,252,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,
        rgba(14,165,233,0.14), rgba(99,102,241,0.14));
    border: 1px solid rgba(14,165,233,0.35);
    color: #7dd3fc;
    border-radius: 10px;
    padding: 0.45rem 1.4rem;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.3px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg,
        rgba(14,165,233,0.26), rgba(99,102,241,0.26));
    border-color: rgba(14,165,233,0.6);
    box-shadow: 0 0 22px rgba(14,165,233,0.18);
    transform: translateY(-1px);
    color: #bae6fd;
}
.stButton > button:active { transform: translateY(0); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus {
    border-color: rgba(14,165,233,0.45) !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.10) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] > section {
    background: rgba(14,165,233,0.04);
    border: 2px dashed rgba(14,165,233,0.28);
    border-radius: 14px;
    transition: all 0.2s;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: rgba(14,165,233,0.55);
    background: rgba(14,165,233,0.08);
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    border-radius: 99px;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px 14px;
}

/* ── Checkbox / Toggle ── */
.stCheckbox label { color: rgba(200,215,255,0.75) !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.10);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.20); }

/* ── Audio input ── */
[data-testid="stAudioInput"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.5rem;
}

/* ── Alert / Info boxes ── */
[data-testid="stAlert"] {
    background: rgba(14,165,233,0.08) !important;
    border: 1px solid rgba(14,165,233,0.2) !important;
    border-radius: 12px !important;
    color: #93c5fd !important;
}

/* ── Code / SRT preview ── */
.srt-block {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.82rem;
    line-height: 1.9;
    color: #94a3b8;
    background: rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 20px;
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    margin-top: 8px;
}

/* ── Glass panel divs ── */
.glass-panel {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.glass-panel-blue {
    background: linear-gradient(135deg,
        rgba(14,165,233,0.07), rgba(99,102,241,0.07));
    border: 1px solid rgba(14,165,233,0.15);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 14px;
}

/* ── Transcript line ── */
.tx-line {
    padding: 10px 14px;
    margin: 6px 0;
    background: rgba(255,255,255,0.03);
    border-left: 3px solid rgba(14,165,233,0.5);
    border-radius: 0 10px 10px 0;
    color: #cbd5e1;
    font-size: 0.93rem;
    line-height: 1.5;
}
.tx-time {
    font-size: 0.75rem;
    color: rgba(148,163,184,0.6);
    margin-bottom: 2px;
    font-family: 'Consolas', monospace;
}
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════

def _srt_ts(s: float) -> str:
    ms = int(round(s * 1000))
    hh = ms // 3_600_000; ms %= 3_600_000
    mm = ms // 60_000;    ms %= 60_000
    ss = ms // 1_000;     ms %= 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _audio_bytes_to_np(audio_bytes: bytes) -> np.ndarray | None:
    """將 st.audio_input 回傳的 bytes（WAV）轉為 16kHz float32 array。"""
    try:
        import librosa
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        audio, _ = librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
        os.unlink(tmp)
        return audio
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
# 模型載入（@st.cache_resource：全域單例，所有 session 共用）
# ══════════════════════════════════════════════════════════

def _load_settings() -> dict:
    """讀取 settings.json，回傳設定 dict（失敗時回傳空 dict）。"""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(d: dict) -> None:
    """將設定 dict 寫入 settings.json。"""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


@st.cache_resource(show_spinner=False)
def _load_engine():
    """載入 ChatLLMASREngine（只執行一次，所有 session 共用同一實例）。
    回傳 (engine, error_msg)。
    """
    try:
        # 讀取 settings.json
        settings = _load_settings()
        model_path  = settings.get("model_path")
        chatllm_dir = settings.get("chatllm_dir")

        if not model_path:
            return None, "settings.json 缺少 model_path，請先完成初始設定"
        if not Path(model_path).exists():
            return None, f"找不到模型檔案：{model_path}"
        if not chatllm_dir or not Path(chatllm_dir).exists():
            return None, f"找不到 chatllm 目錄：{chatllm_dir}"

        # 確保 chatllm_engine.py 可 import（與 streamlit_app.py 同層）
        engine_dir = str(BASE_DIR)
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)

        from chatllm_engine import ChatLLMASREngine

        engine = ChatLLMASREngine()
        engine.load(
            model_path   = model_path,
            chatllm_dir  = chatllm_dir,
            n_gpu_layers = 99,
        )
        return engine, None

    except Exception as e:
        import traceback
        return None, f"{e}\n{traceback.format_exc()}"


def _transcribe(engine, audio: np.ndarray, language=None, context=None) -> str:
    """audio: 16kHz float32 numpy array → 轉錄文字（已含繁體轉換）。"""
    return engine.transcribe(audio, language=language, context=context)


def _process_file(engine, audio_path: Path,
                  language=None, context=None,
                  diarize=False, n_speakers=None,
                  progress_cb=None) -> str | None:
    """呼叫 engine.process_file()，將 SRT 內容以字串回傳。
    progress_cb(pct: float, msg: str) — pct 為 0.0~1.0。
    """
    def _adapted_cb(i, total, msg):
        if progress_cb and total > 0:
            progress_cb(i / total, msg)

    out_path = engine.process_file(
        audio_path,
        progress_cb = _adapted_cb if progress_cb else None,
        language    = language,
        context     = context,
        diarize     = diarize,
        n_speakers  = n_speakers,
    )
    if out_path and out_path.exists():
        return out_path.read_text(encoding="utf-8")
    return None


# ══════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════

def _render_sidebar(engine, err: str | None):
    with st.sidebar:
        st.markdown("""
<div style="text-align:center; padding: 8px 0 16px;">
  <div style="font-size:2rem;">🎙</div>
  <div style="font-size:1.15rem; font-weight:700;
              background: linear-gradient(90deg, #7dd3fc, #a5b4fc);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    Qwen3 ASR
  </div>
  <div style="font-size:0.72rem; color:rgba(148,163,184,0.6);
              margin-top:2px; letter-spacing:0.5px;">
    VULKAN GPU FRONTEND
  </div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # 模型狀態
        if err:
            st.markdown(f"""
<div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.25);
            border-radius:12px; padding:12px 14px; margin-bottom:12px;">
  <div style="color:#f87171; font-weight:600; font-size:0.85rem;">❌ 模型載入失敗</div>
  <div style="color:rgba(248,113,113,0.7); font-size:0.75rem;
              margin-top:4px; word-break:break-all;">{err[:300]}</div>
</div>""", unsafe_allow_html=True)
        elif engine and engine.ready:
            # 從 settings.json 讀取裝置資訊（不需要 torch）
            settings  = _load_settings()
            dev_str   = settings.get("device", "Vulkan GPU")
            use_dll   = getattr(engine, "_use_dll", False)
            mode_str  = "DLL 常駐" if use_dll else "Subprocess"
            st.markdown(f"""
<div style="background:rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.2);
            border-radius:12px; padding:12px 14px; margin-bottom:12px;">
  <div style="color:#4ade80; font-weight:700; font-size:0.85rem;">
    ✅ 就緒 · Vulkan
  </div>
  <div style="color:rgba(148,163,184,0.7); font-size:0.73rem; margin-top:5px;">
    Qwen3-ASR-1.7B .bin
  </div>
  <div style="color:rgba(148,163,184,0.55); font-size:0.68rem; margin-top:3px;">
    ⚡ {dev_str[:40]}<br>
    &nbsp;&nbsp;&nbsp;模式：{mode_str}
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.2);
            border-radius:12px; padding:12px 14px; margin-bottom:12px;">
  <div style="color:#fbbf24; font-weight:600; font-size:0.85rem;">⏳ 載入中…</div>
</div>""", unsafe_allow_html=True)

        st.divider()

        # 全域設定
        st.markdown('<div style="color:rgba(148,163,184,0.7); font-size:0.8rem; '
                    'font-weight:600; letter-spacing:0.5px; margin-bottom:8px;">'
                    'GLOBAL SETTINGS</div>', unsafe_allow_html=True)

        lang_options = ["自動偵測"] + SUPPORTED_LANGUAGES
        language = st.selectbox(
            "辨識語系",
            options=lang_options,
            index=0,
            help="強制指定辨識語言，自動偵測適合多語混合音訊",
        )

        st.divider()

        # 說話者分離
        st.markdown('<div style="color:rgba(148,163,184,0.7); font-size:0.8rem; '
                    'font-weight:600; letter-spacing:0.5px; margin-bottom:8px;">'
                    'DIARIZATION</div>', unsafe_allow_html=True)

        diar_available = (engine is not None
                          and getattr(engine, "diar_engine", None) is not None)
        diarize = st.checkbox(
            "說話者分離",
            value=False,
            disabled=not diar_available,
            help="標記不同說話者（僅限音檔模式）" if diar_available
                 else "需要說話者分離模型（diarization/）",
        )
        n_speakers = None
        if diarize and diar_available:
            spk_sel = st.select_slider(
                "說話人數",
                options=["自動", "2", "3", "4", "5", "6", "7", "8"],
                value="自動",
            )
            n_speakers = int(spk_sel) if spk_sel != "自動" else None

        st.divider()
        st.markdown(
            '<div style="color:rgba(100,116,139,0.5); font-size:0.7rem; '
            'text-align:center; line-height:1.6;">'
            'Qwen3-ASR-1.7B · chatllm + Vulkan<br>silero VAD · OpenCC s2twp'
            '</div>', unsafe_allow_html=True
        )

    return language if language != "自動偵測" else None, diarize, n_speakers


# ══════════════════════════════════════════════════════════
# Tab 1 — 音檔轉字幕
# ══════════════════════════════════════════════════════════

def _tab_file(engine):
    st.markdown("""
<div class="glass-panel-blue">
  <div style="font-size:1.05rem; font-weight:700; color:#7dd3fc; margin-bottom:4px;">
    🎵 音檔轉字幕
  </div>
  <div style="font-size:0.82rem; color:rgba(148,163,184,0.6);">
    上傳音訊檔案，AI 自動分段辨識並輸出 SRT 字幕
  </div>
</div>""", unsafe_allow_html=True)

    # 上傳
    uploaded = st.file_uploader(
        "拖曳或選擇音訊檔案",
        type=["mp3", "wav", "flac", "m4a", "ogg", "aac"],
        help="支援 MP3 / WAV / FLAC / M4A / OGG / AAC",
    )

    # 辨識提示
    hint = st.text_area(
        "辨識提示（可選）",
        placeholder="貼入歌詞、關鍵字或背景說明，可提升辨識準確度…",
        height=80,
        help="Context 會插入 system message，引導模型辨識特定詞彙",
    )

    # 轉換按鈕
    ready = engine is not None and uploaded is not None
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        convert = st.button(
            "▶  開始轉換",
            disabled=not ready,
            type="primary",
            use_container_width=True,
        )
    with col_info:
        if not engine:
            st.markdown(
                '<div style="color:#fbbf24; font-size:0.83rem; '
                'padding-top:8px;">⏳ 等待模型載入完成</div>',
                unsafe_allow_html=True
            )
        elif not uploaded:
            st.markdown(
                '<div style="color:rgba(148,163,184,0.5); font-size:0.83rem; '
                'padding-top:8px;">← 請先上傳音訊檔案</div>',
                unsafe_allow_html=True
            )

    # ── 執行轉換 ──
    if convert and ready:
        lang  = st.session_state.get("_lang")
        diar  = st.session_state.get("_diar", False)
        n_spk = st.session_state.get("_nspk")
        context = hint.strip() or None

        prog_bar   = st.progress(0.0)
        prog_label = st.empty()

        def _cb(pct, msg):
            prog_bar.progress(pct)
            prog_label.markdown(
                f'<div style="color:rgba(148,163,184,0.7); font-size:0.82rem;">'
                f'{msg}</div>', unsafe_allow_html=True
            )

        with tempfile.NamedTemporaryFile(
            suffix="." + uploaded.name.rsplit(".", 1)[-1], delete=False
        ) as f:
            f.write(uploaded.read())
            tmp_path = Path(f.name)

        try:
            t0 = time.perf_counter()
            _cb(0.02, "音訊載入中…")
            srt_content = _process_file(
                engine, tmp_path,
                language=lang, context=context,
                diarize=diar, n_speakers=n_spk,
                progress_cb=_cb,
            )
            elapsed = time.perf_counter() - t0
            prog_bar.progress(1.0)

            if srt_content:
                st.session_state["srt_content"]  = srt_content
                st.session_state["srt_filename"]  = uploaded.name.rsplit(".", 1)[0] + ".srt"
                st.session_state["srt_elapsed"]   = elapsed
                prog_label.markdown(
                    f'<div style="color:#4ade80; font-size:0.85rem; font-weight:600;">'
                    f'✅ 完成！耗時 {elapsed:.1f}s</div>',
                    unsafe_allow_html=True
                )
            else:
                prog_label.markdown(
                    '<div style="color:#fbbf24; font-size:0.85rem;">'
                    '⚠ 未偵測到人聲，無字幕產生</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            prog_label.markdown(
                f'<div style="color:#f87171; font-size:0.85rem;">❌ 錯誤：{e}</div>',
                unsafe_allow_html=True
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── 結果顯示 ──
    srt = st.session_state.get("srt_content")
    if srt:
        st.divider()
        fname   = st.session_state.get("srt_filename", "output.srt")
        elapsed = st.session_state.get("srt_elapsed", 0)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f'<div style="color:#7dd3fc; font-weight:700; font-size:1rem;">'
                f'📄 {fname}</div>'
                f'<div style="color:rgba(148,163,184,0.5); font-size:0.75rem;">'
                f'{len(srt.strip().splitlines())} 行 · {elapsed:.1f}s</div>',
                unsafe_allow_html=True
            )
        with c2:
            st.download_button(
                label="⬇ 下載 SRT",
                data=srt.encode("utf-8"),
                file_name=fname,
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown(
            f'<div class="srt-block">{srt[:3000]}'
            f'{"…（僅顯示前段）" if len(srt) > 3000 else ""}</div>',
            unsafe_allow_html=True,
        )

        if st.button("🗑 清除結果", use_container_width=False):
            for k in ("srt_content", "srt_filename", "srt_elapsed"):
                st.session_state.pop(k, None)
            st.rerun()


# ══════════════════════════════════════════════════════════
# Tab 2 — 即時辨識
# ══════════════════════════════════════════════════════════

def _tab_realtime(engine):
    st.markdown("""
<div class="glass-panel-blue">
  <div style="font-size:1.05rem; font-weight:700; color:#7dd3fc; margin-bottom:4px;">
    🎙 即時語音辨識
  </div>
  <div style="font-size:0.82rem; color:rgba(148,163,184,0.6);">
    按下麥克風錄音 → 放開後自動辨識，可持續累積字幕
  </div>
</div>""", unsafe_allow_html=True)

    col_mic, col_hint = st.columns([1, 2])

    with col_hint:
        rt_hint = st.text_input(
            "辨識提示（可選）",
            placeholder="歌詞、關鍵字或背景說明…",
            help="引導模型辨識特定詞彙",
        )

    with col_mic:
        audio_data = st.audio_input(
            "點此錄音",
            disabled=engine is None,
        )

    # ── 自動處理錄音 ──
    if audio_data is not None and engine is not None:
        lang    = st.session_state.get("_lang")
        context = rt_hint.strip() or None

        with st.spinner("辨識中…"):
            audio_np = _audio_bytes_to_np(audio_data.getvalue())

        if audio_np is not None and len(audio_np) >= SAMPLE_RATE * 0.5:
            try:
                text = _transcribe(engine, audio_np, language=lang, context=context)
                if text:
                    ts = datetime.now().strftime("%H:%M:%S")
                    if "rt_log" not in st.session_state:
                        st.session_state["rt_log"] = []
                    st.session_state["rt_log"].append((ts, text))
            except Exception as e:
                st.error(f"辨識失敗：{e}")
        else:
            st.warning("錄音太短，請再錄一次（至少 0.5 秒）")

    # ── 記錄顯示 ──
    log: list[tuple[str, str]] = st.session_state.get("rt_log", [])

    col_act1, col_act2, col_act3 = st.columns([1, 1, 3])
    with col_act1:
        if st.button("🗑 清除", use_container_width=True, disabled=not log):
            st.session_state["rt_log"] = []
            st.rerun()
    with col_act2:
        if log:
            # 組成 SRT
            srt_lines = []
            for idx, (ts, text) in enumerate(log, 1):
                start = (idx - 1) * 5.0
                end   = start + 5.0
                srt_lines.append(f"{idx}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{text}\n")
            srt_bytes = "\n".join(srt_lines).encode("utf-8")
            ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "💾 下載 SRT",
                data=srt_bytes,
                file_name=f"realtime_{ts_now}.srt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()

    if not log:
        st.markdown("""
<div style="text-align:center; padding: 40px 0; color:rgba(100,116,139,0.5);">
  <div style="font-size:2.5rem; margin-bottom:8px;">🎤</div>
  <div style="font-size:0.85rem;">按下上方麥克風開始錄音</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="color:rgba(148,163,184,0.5); font-size:0.75rem; '
            f'margin-bottom:8px;">共 {len(log)} 段</div>',
            unsafe_allow_html=True,
        )
        # 最新的排最上面
        for ts, text in reversed(log):
            st.markdown(f"""
<div class="tx-line">
  <div class="tx-time">{ts}</div>
  <div>{text}</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# Tab 3 — 設定
# ══════════════════════════════════════════════════════════

def _download_model_ui(target_path: str) -> None:
    """在 Streamlit UI 中以串流下載模型，即時更新進度條。"""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    bar  = st.progress(0.0, text="正在建立連線…")
    info = st.empty()

    try:
        import requests
        resp = requests.get(MODEL_DOWNLOAD_URL, stream=True, timeout=30)
        resp.raise_for_status()

        total      = int(resp.headers.get("content-length", 0))
        downloaded = 0
        t0         = time.time()

        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = max(time.time() - t0, 0.001)
                    speed   = downloaded / elapsed / 1024 / 1024
                    if total > 0:
                        pct      = downloaded / total
                        mb_done  = downloaded / 1024 / 1024
                        mb_total = total / 1024 / 1024
                        bar.progress(
                            pct,
                            text=f"下載中… {mb_done:.0f} / {mb_total:.0f} MB"
                                 f"  （{speed:.1f} MB/s）",
                        )

        bar.progress(1.0, text="✅ 下載完成！")
        info.success(f"模型已儲存至：{target}")
        time.sleep(1.5)
        st.rerun()

    except Exception as e:
        bar.empty()
        st.error(f"下載失敗：{e}")


def _tab_settings() -> None:
    settings = _load_settings()

    st.markdown("""
<div class="glass-panel-blue">
  <div style="font-size:1.05rem; font-weight:700; color:#7dd3fc; margin-bottom:4px;">
    ⚙️ 服務設定
  </div>
  <div style="font-size:0.82rem; color:rgba(148,163,184,0.6);">
    設定推理裝置與模型路徑，儲存後自動重新載入引擎
  </div>
</div>""", unsafe_allow_html=True)

    # ── 1. chatllm 目錄 + GPU 偵測 ──────────────────────────────────
    st.markdown(
        '<div style="color:rgba(148,163,184,0.7); font-size:0.8rem; '
        'font-weight:600; letter-spacing:0.5px; margin-bottom:8px;">'
        'INFERENCE BACKEND</div>',
        unsafe_allow_html=True,
    )

    col_dir, col_btn = st.columns([4, 1])
    with col_dir:
        chatllm_dir = st.text_input(
            "chatllm 目錄（含 main.exe / libchatllm.dll）",
            value=settings.get("chatllm_dir", str(BASE_DIR / "chatllm")),
            placeholder=str(BASE_DIR / "chatllm"),
        )
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🔍 偵測 GPU", use_container_width=True):
            if not Path(chatllm_dir).exists():
                st.error(f"目錄不存在：{chatllm_dir}")
            else:
                with st.spinner("執行 main.exe --show_devices…"):
                    try:
                        if str(BASE_DIR) not in sys.path:
                            sys.path.insert(0, str(BASE_DIR))
                        from chatllm_engine import detect_vulkan_devices
                        devs = detect_vulkan_devices(chatllm_dir)
                        st.session_state["_detected_gpus"] = devs
                        if devs:
                            st.success(f"偵測到 {len(devs)} 個 Vulkan GPU")
                        else:
                            st.warning("未偵測到 Vulkan GPU，將使用 CPU 模式")
                    except Exception as e:
                        st.error(f"偵測失敗：{e}")

    # 裝置下拉
    detected: list[dict] | None = st.session_state.get("_detected_gpus")
    if detected is None:
        # 尚未偵測：從 settings 還原目前裝置選項
        cur_dev = settings.get("device", "CPU")
        dev_options = ["CPU"]
        if cur_dev and cur_dev != "CPU":
            dev_options.append(cur_dev)
    else:
        dev_options = ["CPU"] + [
            f"GPU:{d['id']} ({d['name']}) [Vulkan]" for d in detected
        ]

    cur_dev = settings.get("device", "CPU")
    try:
        dev_idx = dev_options.index(cur_dev)
    except ValueError:
        dev_idx = 0

    selected_device = st.selectbox(
        "推理裝置",
        options=dev_options,
        index=dev_idx,
        help="CPU → OpenVINO   |   GPU (Vulkan) → chatllm + GGUF .bin",
    )

    # VRAM 資訊提示
    if detected:
        for d in detected:
            if f"GPU:{d['id']} ({d['name']}) [Vulkan]" == selected_device:
                vram_gb = d["vram_free"] / 1024 ** 3
                st.caption(f"🎮 可用 VRAM：{vram_gb:.1f} GB（模型約需 2.3 GB）")

    st.divider()

    # ── 2. 模型路徑 ─────────────────────────────────────────────────
    st.markdown(
        '<div style="color:rgba(148,163,184,0.7); font-size:0.8rem; '
        'font-weight:600; letter-spacing:0.5px; margin-bottom:8px;">'
        'MODEL</div>',
        unsafe_allow_html=True,
    )

    default_model = settings.get(
        "model_path", str(BASE_DIR / "GPUModel" / "qwen3-asr-1.7b.bin")
    )
    model_path = st.text_input(
        "模型路徑（.bin）",
        value=default_model,
        placeholder=str(BASE_DIR / "GPUModel" / "qwen3-asr-1.7b.bin"),
    )

    if model_path:
        mp = Path(model_path)
        if mp.exists():
            size_mb = mp.stat().st_size // (1024 * 1024)
            st.markdown(
                f'<div style="background:rgba(74,222,128,0.08); '
                f'border:1px solid rgba(74,222,128,0.2); border-radius:10px; '
                f'padding:10px 14px; font-size:0.82rem; color:#4ade80;">'
                f'✅ 已找到模型 · {size_mb:,} MB</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(251,191,36,0.08); '
                'border:1px solid rgba(251,191,36,0.2); border-radius:10px; '
                'padding:10px 14px; font-size:0.82rem; color:#fbbf24;">'
                '⚠️ 模型檔案不存在，可點選下方按鈕下載</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"下載來源：{MODEL_DOWNLOAD_URL}")
            if st.button("⬇ 下載模型（約 2.3 GB）", type="secondary"):
                _download_model_ui(model_path)

    st.divider()

    # ── 3. 儲存設定 ─────────────────────────────────────────────────
    col_save, col_info = st.columns([1, 2])

    with col_save:
        if st.button(
            "💾 儲存設定並重新載入引擎",
            type="primary",
            use_container_width=True,
        ):
            _save_settings({
                **settings,
                "backend":     "chatllm",
                "device":      selected_device,
                "model_path":  model_path,
                "chatllm_dir": chatllm_dir,
            })
            st.cache_resource.clear()
            st.success("✅ 設定已儲存，引擎將在頁面重新整理後載入")
            time.sleep(1.0)
            st.rerun()

    # ── 4. 當前生效設定（唯讀顯示）─────────────────────────────────
    with col_info:
        active = _load_settings()
        if active:
            mp_name = Path(active.get("model_path", "")).name or "—"
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.03); '
                f'border:1px solid rgba(255,255,255,0.07); '
                f'border-radius:10px; padding:10px 14px; '
                f'font-size:0.75rem; color:rgba(148,163,184,0.65); line-height:1.8;">'
                f'<b style="color:rgba(148,163,184,0.9);">當前生效設定</b><br>'
                f'後端：{active.get("backend", "—")}<br>'
                f'裝置：{active.get("device", "—")[:50]}<br>'
                f'模型：{mp_name}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════

def main():
    # ── 載入引擎（spinner 只在第一次顯示）──
    with st.spinner("🔄 載入 Qwen3-ASR 模型…（首次需要約 20–40 秒）"):
        engine, err = _load_engine()

    # ── Sidebar（同時把選項寫入 session_state）──
    language, diarize, n_speakers = _render_sidebar(engine, err)
    st.session_state["_lang"]  = language
    st.session_state["_diar"]  = diarize
    st.session_state["_nspk"]  = n_speakers

    # ── Header ──
    st.markdown("""
<div style="
    background: linear-gradient(135deg,
        rgba(14,165,233,0.10), rgba(99,102,241,0.10));
    border: 1px solid rgba(14,165,233,0.18);
    border-radius: 20px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 16px;
">
  <div style="font-size:2.2rem; line-height:1;">🎙</div>
  <div>
    <div style="
        font-size:1.4rem; font-weight:800; letter-spacing:-0.3px;
        background: linear-gradient(90deg, #7dd3fc 0%, #a5b4fc 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    ">Qwen3 ASR 字幕生成器</div>
    <div style="font-size:0.78rem; color:rgba(148,163,184,0.55);
                margin-top:2px; letter-spacing:0.3px;">
      GPU-Accelerated · Qwen3-ASR-1.7B · chatllm + Vulkan
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs([
        "   🎵  音檔轉字幕   ",
        "   🎙  即時辨識   ",
        "   ⚙️  設定   ",
    ])

    with tab1:
        _tab_file(engine)

    with tab2:
        _tab_realtime(engine)

    with tab3:
        _tab_settings()


if __name__ == "__main__":
    main()
