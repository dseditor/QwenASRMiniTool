# Qwen3 ASR MiniTool

本地語音辨識字幕生成工具。基於 Qwen3-ASR-0.6B 模型，使用 OpenVINO INT8 量化推理，**不需要 NVIDIA GPU**，純 CPU 即可執行。

## 功能

- 音訊檔案（MP3 / WAV / FLAC / M4A / OGG）→ SRT 字幕
- 即時語音辨識（麥克風輸入）
- 自動 VAD 靜音偵測，分段轉錄
- 簡轉繁中文輸出
- 首次執行自動下載模型（約 1.2 GB）

## 模型來源

| 項目 | 連結 |
|------|------|
| OpenVINO INT8 量化版 | [Echo9Zulu/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO](https://huggingface.co/Echo9Zulu/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO) |
| 原始 PyTorch 模型 | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| VAD 模型 | [snakers4/silero-vad v4.0](https://github.com/snakers4/silero-vad) |

## 快速開始

## 系統需求

| 項目 | 最低要求 |
|------|---------|
| 作業系統 | Windows 10 / 11（64-bit）|
| Python | 3.10 以上 |
| RAM | 6 GB（推理時峰值約 4.8 GB）|
| 硬碟空間 | 2 GB（模型 1.2 GB + 程式）|
| CPU | Intel 11th Gen+ 或同等級 AMD |

> GPU 非必要。OpenVINO GPU 外掛僅支援 Intel GPU，不支援 NVIDIA。

## 開發者說明

### 重新產生 prompt_template.json

processor_numpy.py 的 BPE prompt 模板需要從原始模型提取一次。若更換模型版本，請在有 torch + transformers 的環境中執行：

\載入 qwen_asr processor…
這會在 ov_models/ 目錄下產生：
- prompt_template.json：Prompt token ID 模板
- mel_filters.npy：Mel 濾波器矩陣

### 架構說明

## 授權

本專案程式碼以 MIT 授權釋出。模型權重依各自來源的授權條款。
