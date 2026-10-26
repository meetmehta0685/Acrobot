"""Render the saved Acrobot study GIFs as labelled MP4 demonstrations."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "results" / "study" / "demos"
VIDEO_DIR = ROOT / "results" / "study" / "videos"

WIDTH, HEIGHT = 1280, 720
FPS = 15
BACKGROUND = (15, 23, 32)
PANEL = (25, 36, 48)
TEXT = (238, 242, 245)
MUTED = (165, 178, 190)
ACCENTS = {
    "Q-Learning": (59, 130, 246),
    "DQN": (249, 115, 22),
    "PPO": (16, 185, 129),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT = _font(34, bold=True)
LABEL_FONT = _font(24, bold=True)
BODY_FONT = _font(19)
SMALL_FONT = _font(16)


def _centered(draw: ImageDraw.ImageDraw, text: str, y: int, font, fill=TEXT) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((WIDTH - (box[2] - box[0])) // 2, y), text, font=font, fill=fill)


def _load_gif(path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with Image.open(path) as gif:
        for index in range(gif.n_frames):
            gif.seek(index)
            frames.append(gif.convert("RGB").copy())
    return frames


def _resize(frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    return frame.resize(size, Image.Resampling.LANCZOS)


def _comparison_frame(
    frames: dict[str, list[Image.Image]],
    labels: list[str],
    index: int,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _centered(draw, "Acrobot-v1: three learned controllers", 24, TITLE_FONT)
    _centered(
        draw,
        "Greedy held-out demonstrations from the completed study",
        68,
        BODY_FONT,
        MUTED,
    )

    panel_size = 360
    gap = 30
    left = (WIDTH - (3 * panel_size + 2 * gap)) // 2
    top = 145
    for position, label in enumerate(labels):
        x = left + position * (panel_size + gap)
        accent = ACCENTS[label]
        draw.rounded_rectangle(
            (x - 4, top - 4, x + panel_size + 4, top + panel_size + 4),
            radius=12,
            fill=accent,
        )
        draw.rounded_rectangle(
            (x, top, x + panel_size, top + panel_size),
            radius=9,
            fill=PANEL,
        )
        source = frames[label]
        frame = source[min(index, len(source) - 1)]
        image.paste(_resize(frame, (panel_size, panel_size)), (x, top))
        draw.text((x, top - 38), label, font=LABEL_FONT, fill=accent)
        if index >= len(source) - 1:
            done_text = "episode ended"
            box = draw.textbbox((0, 0), done_text, font=SMALL_FONT)
            draw.rounded_rectangle(
                (x + panel_size - (box[2] - box[0]) - 16, top + 12,
                 x + panel_size - 8, top + 38),
                radius=8,
                fill=(0, 0, 0),
            )
            draw.text(
                (x + panel_size - (box[2] - box[0]) - 12, top + 16),
                done_text,
                font=SMALL_FONT,
                fill=TEXT,
            )

    draw.line((40, 650, WIDTH - 40, 650), fill=(57, 72, 87), width=1)
    _centered(
        draw,
        "Each panel uses the strongest completed study checkpoint for that method.",
        668,
        SMALL_FONT,
        MUTED,
    )
    return image


def _title_frame() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _centered(draw, "Acrobot reinforcement learning", 230, TITLE_FONT)
    _centered(draw, "Q-Learning  ·  DQN  ·  PPO", 290, LABEL_FONT, (105, 211, 178))
    _centered(draw, "Three controllers solving the same underactuated swing-up task", 345, BODY_FONT, MUTED)
    _centered(draw, "Gymnasium Acrobot-v1", 390, SMALL_FONT, MUTED)
    return image


def _summary_frame() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _centered(draw, "Full study summary", 70, TITLE_FONT)
    _centered(draw, "100,000 interactions per run · 5 final seeds", 120, BODY_FONT, MUTED)

    columns = (250, 600, 850, 1050)
    headers = ("Method", "Final return IQM", "Success IQM", "Mean time")
    for x, header in zip(columns, headers):
        draw.text((x, 210), header, font=LABEL_FONT, fill=TEXT)
    rows = (
        ("PPO", "-81.7", "100%", "30.5 s"),
        ("DQN", "-434.7", "24%", "18.8 s"),
        ("Q-Learning", "-480.5", "17%", "6.6 s"),
    )
    for row_index, row in enumerate(rows):
        y = 285 + row_index * 76
        accent = ACCENTS[row[0]]
        draw.rounded_rectangle((180, y - 10, 1110, y + 48), radius=8, fill=PANEL)
        for x, value in zip(columns, row):
            draw.text((x, y), value, font=LABEL_FONT, fill=accent if x == columns[0] else TEXT)
    _centered(draw, "Higher return is better because Acrobot rewards are negative step costs.", 555, BODY_FONT, MUTED)
    _centered(draw, "The comparison is conditional on this implementation, search space, seeds, and budget.", 600, SMALL_FONT, MUTED)
    return image


def _write_video(path: Path, frames: list[Image.Image]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def render_individual(label: str, frames: list[Image.Image]) -> Path:
    output = VIDEO_DIR / f"{label.lower().replace('-', '_')}_demo.mp4"
    title = _title_frame()
    video_frames = [title] * FPS
    for frame in frames:
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        _centered(draw, f"Acrobot-v1 · {label}", 28, TITLE_FONT)
        _centered(draw, "Greedy policy at the strongest completed study checkpoint", 72, BODY_FONT, MUTED)
        x, y, size = (WIDTH - 500) // 2, 135, 500
        draw.rounded_rectangle((x - 4, y - 4, x + size + 4, y + size + 4), radius=12, fill=ACCENTS[label])
        canvas.paste(_resize(frame, (size, size)), (x, y))
        _centered(draw, "Held-out reset seed 90,000", 665, SMALL_FONT, MUTED)
        video_frames.append(canvas)
    video_frames.extend([video_frames[-1]] * (FPS // 2))
    _write_video(output, video_frames)
    return output


def main() -> None:
    source_names = {
        "Q-Learning": "q_learning_demo.gif",
        "DQN": "dqn_demo.gif",
        "PPO": "ppo_demo.gif",
    }
    frames = {label: _load_gif(DEMO_DIR / filename) for label, filename in source_names.items()}
    labels = list(source_names)
    playback = [
        _comparison_frame(frames, labels, index)
        for index in range(max(len(items) for items in frames.values()))
    ]
    comparison = [_title_frame()] * FPS
    comparison.extend(playback)
    comparison.extend([_summary_frame()] * (FPS * 3))
    comparison_path = VIDEO_DIR / "acrobot_algorithm_comparison.mp4"
    _write_video(comparison_path, comparison)
    print(f"Wrote {comparison_path}")
    for label in labels:
        print(f"Wrote {render_individual(label, frames[label])}")


if __name__ == "__main__":
    main()
