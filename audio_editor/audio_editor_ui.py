import os
import time
from typing import List, Tuple

import sys

import gradio as gr

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.append(_TOOLS_DIR)

from audio_editor import _apply_speed, _concat_audios, _insert_silences


def _normalize_file_paths(files) -> List[str]:
    if not files:
        return []
    paths = []
    for item in files:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict) and "name" in item:
            paths.append(item["name"])
        elif hasattr(item, "name"):
            paths.append(item.name)
        else:
            raise ValueError(f"Unsupported file input: {item!r}")
    return paths


def _parse_silence_table(rows) -> List[Tuple[float, float]]:
    if rows is None:
        return []
    if hasattr(rows, "empty") and rows.empty:
        return []
    if hasattr(rows, "values"):
        rows = rows.values.tolist()

    if not rows:
        return []

    silences = []
    for row in rows:
        if not row or len(row) < 2:
            continue
        time_val = row[0]
        dur_val = row[1]
        if time_val is None or dur_val is None:
            continue
        time_val = float(time_val)
        dur_val = float(dur_val)
        if time_val < 0 or dur_val < 0:
            raise ValueError("Silence time/duration must be >= 0.")
        silences.append((time_val, dur_val))
    return silences


def edit_audio(files, speed, preserve_pitch, silence_rows):
    paths = _normalize_file_paths(files)
    if not paths:
        raise ValueError("Please upload at least one audio file.")

    wav, sr = _concat_audios(paths)
    wav, sr = _apply_speed(wav, sr, float(speed), bool(preserve_pitch))

    silences = _parse_silence_table(silence_rows)
    wav = _insert_silences(wav, sr, silences)

    output_dir = os.path.join("outputs", "audio_editor")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"edited_{int(time.time())}.wav")

    from audio_editor import _save_audio

    _save_audio(output_path, wav, sr)
    duration = wav.shape[1] / sr
    info = f"Saved: {output_path} ({duration:.2f}s, {sr}Hz)"

    return output_path, output_path, info


def preview_concat(files, speed, preserve_pitch, silence_rows):
    paths = _normalize_file_paths(files)
    if not paths:
        raise ValueError("请先上传音频文件。")

    wav, sr = _concat_audios(paths)
    wav, sr = _apply_speed(wav, sr, float(speed), bool(preserve_pitch))
    silences = _parse_silence_table(silence_rows)
    wav = _insert_silences(wav, sr, silences)
    output_dir = os.path.join("outputs", "audio_editor")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"preview_{int(time.time())}.wav")

    from audio_editor import _save_audio

    _save_audio(output_path, wav, sr)
    duration = wav.shape[1] / sr
    info = f"拼接预览：{duration:.2f}s, {sr}Hz（已应用静音）"
    return output_path, info, gr.update(maximum=max(0.01, duration), value=0.0)


def add_silence_row(rows, insert_time, silence_duration):
    if rows is None:
        rows = []
    elif hasattr(rows, "values"):
        rows = rows.values.tolist()

    if not rows:
        rows = []
    if insert_time is None or silence_duration is None:
        return rows
    rows = list(rows)
    rows.append([float(insert_time), float(silence_duration)])
    return rows


with gr.Blocks(title="音频编辑器") as demo:
    gr.Markdown(
        "## 音频编辑器\n"
        "\n"
        "### 使用说明\n"
        "1. 上传一个或多个音频文件，顺序即拼接顺序。\n"
        "2. 设置播放速度：大于 1 加速，小于 1 减速。\n"
        "3. 可勾选“保持音高”以降低变速导致的音高变化（需要安装 `librosa`）。\n"
        "4. 在“插入静音”表格中填写插入点与时长（秒）。可填写多行。\n"
        "5. 点击“开始处理”生成并播放/下载结果。\n"
        "\n"
        "说明：插入静音的时间点基于原始拼接后的时间轴，\n"
        "每次插入会使后续插入点自动后移。"
    )

    with gr.Row():
        files = gr.File(
            label="输入音频文件（顺序即拼接顺序）",
            file_types=["audio"],
            file_count="multiple",
        )
        output_audio = gr.Audio(label="输出音频", type="filepath")

    with gr.Row():
        speed = gr.Slider(
            label="播放速度",
            minimum=0.25,
            maximum=3.0,
            value=1.0,
            step=0.05,
        )
        preserve_pitch = gr.Checkbox(
            label="保持音高（需要 librosa）",
            value=False,
        )

    with gr.Row():
        preview_btn = gr.Button("预览拼接")
        preview_audio = gr.Audio(label="拼接预览", type="filepath")
        preview_info = gr.Textbox(label="预览信息", interactive=False)

    silence_table = gr.Dataframe(
        headers=["插入时间（秒）", "静音时长（秒）"],
        datatype=["number", "number"],
        row_count=5,
        col_count=(2, "fixed"),
        label="插入静音（可多行）",
    )

    with gr.Row():
        insert_time = gr.Number(label="插入时间（秒）", value=0.0, minimum=0.0)
        silence_duration = gr.Number(label="静音时长（秒）", value=0.5, minimum=0.0)
        add_row_btn = gr.Button("添加到列表")

    with gr.Row():
        run_btn = gr.Button("开始处理")
        download = gr.File(label="下载输出")
        status = gr.Textbox(label="状态", interactive=False)

    preview_btn.click(
        preview_concat,
        inputs=[files, speed, preserve_pitch, silence_table],
        outputs=[preview_audio, preview_info, insert_time],
    )

    add_row_btn.click(
        add_silence_row,
        inputs=[silence_table, insert_time, silence_duration],
        outputs=[silence_table],
    )

    run_btn.click(
        edit_audio,
        inputs=[files, speed, preserve_pitch, silence_table],
        outputs=[output_audio, download, status],
    )


if __name__ == "__main__":
    demo.launch()
