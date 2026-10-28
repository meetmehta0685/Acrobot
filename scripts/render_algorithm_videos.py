"""Render one demonstration video per algorithm from the saved main-study models.

Each video plays held-out greedy episodes live from the stored checkpoint and
draws the agent's decision values beside the environment:

- Q-Learning: the three table entries Q(s, a) of the current discretized cell;
- DQN: the three network outputs Q_theta(s, a);
- PPO: the policy probabilities pi(a | s) and the critic value V(s).

The free end's height is h = -cos(theta1) - cos(theta1 + theta2); an episode
terminates successfully when h > 1. Using the observation
[cos t1, sin t1, cos t2, sin t2, ...], cos(t1 + t2) = cos t1 cos t2 - sin t1 sin t2.

Every number shown comes from the stored run records or from the episode being
played. Run with:  uv run python scripts/render_algorithm_videos.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import DQNAgent, PPOAgent, QLearningAgent  # noqa: E402
from analysis.stats import iqm  # noqa: E402
from config import DQNConfig, PPOConfig, QLearningConfig  # noqa: E402
from experiments.results import RESULTS_ROOT, load_group  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

VIDEO_DIR = RESULTS_ROOT / "videos"
WIDTH, HEIGHT = 1280, 720
FPS = 30  # one environment step per frame; Acrobot's dt is 0.2 s, so playback is 6x real time
EPISODE_SEEDS = (90_000, 90_001, 90_002)  # first three held-out evaluation seeds
SOLVED_RETURN = -100.0

BACKGROUND = (15, 23, 32)
PANEL = (27, 38, 51)
GRID = (57, 72, 87)
TEXT = (238, 242, 245)
MUTED = (160, 174, 188)
GOOD = (52, 211, 153)
WARN = (251, 191, 36)
BAD = (248, 113, 113)
ACCENTS = {"q_learning": (59, 130, 246), "dqn": (249, 115, 22), "ppo": (16, 185, 129)}
NAMES = {"q_learning": "Q-Learning", "dqn": "DQN", "ppo": "PPO"}
SUMMARY_NAMES = {"q_learning": "Q-Learning", "dqn": "DQN", "ppo": "PPO"}
TORQUES = ("-1", "0", "+1")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    for folder in ("/System/Library/Fonts/Supplemental", "/Library/Fonts"):
        path = Path(folder) / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


H1, H2, BODY, SMALL, MONO = _font(40, True), _font(26, True), _font(21), _font(17), _font(19)


def _centered(draw: ImageDraw.ImageDraw, text: str, y: int, font, fill=TEXT) -> None:
    width = draw.textlength(text, font=font)
    draw.text(((WIDTH - width) / 2, y), text, font=font, fill=fill)


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    return image, ImageDraw.Draw(image)


# ---------------------------------------------------------------- agents

def _params(data: dict) -> dict:
    converted = dict(data)
    for key in ("bins", "hidden_sizes"):
        if key in converted:
            converted[key] = tuple(converted[key])
    return converted


class Controller:
    """Wraps a saved agent: returns the greedy action and the values behind it."""

    def __init__(self, result) -> None:
        self.method = result.method
        params = _params(result.spec["params"])
        checkpoint = RESULTS_ROOT / "models" / result.key
        seed_everything(result.seed)
        if self.method == "q_learning":
            self.agent = QLearningAgent(replace(QLearningConfig(), **params), 3, result.seed)
            self.agent.load(checkpoint.with_suffix(".npz"))
        elif self.method == "dqn":
            self.agent = DQNAgent(replace(DQNConfig(), **params), 6, 3, result.seed, torch.device("cpu"))
            self.agent.load(checkpoint.with_suffix(".pt"))
        else:
            self.agent = PPOAgent(replace(PPOConfig(), **params), 6, 3, result.seed, torch.device("cpu"))
            self.agent.load(checkpoint.with_suffix(".pt"))

    @torch.no_grad()
    def decide(self, observation: np.ndarray) -> tuple[int, dict]:
        if self.method == "q_learning":
            cell = self.agent.discretizer.encode(observation)
            values = self.agent.q_table[cell].astype(float)
            # Same greedy rule as evaluation: arg-max, ties go to the first action.
            return int(np.argmax(values)), {
                "values": values, "cell": cell, "visited": bool(np.any(values != 0.0)),
            }
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        if self.method == "dqn":
            values = self.agent.online(tensor).squeeze(0).numpy().astype(float)
            return int(np.argmax(values)), {"values": values}
        distribution = self.agent.network.distribution(tensor)
        probabilities = distribution.probs.squeeze(0).numpy().astype(float)
        value = float(self.agent.network.value(tensor).item())
        return int(np.argmax(probabilities)), {"values": probabilities, "state_value": value}


def tip_height(observation: np.ndarray) -> float:
    cos1, sin1, cos2, sin2 = observation[:4]
    return float(-cos1 - (cos1 * cos2 - sin1 * sin2))


# ---------------------------------------------------------------- cards

def _description(method: str, params: dict) -> list[str]:
    if method == "q_learning":
        bins = params["bins"]
        cells = int(np.prod(bins))
        return [
            f"Cuts the 6 continuous observations into {'x'.join(map(str, bins))} bins",
            f"= {cells:,} table cells, each holding 3 action values Q(s, a).",
            "Update:  Q(s,a) <- Q(s,a) + alpha [ r + gamma max Q(s',a') - Q(s,a) ]",
            f"alpha = {params['learning_rate']},  gamma = {params['gamma']}.  Acts greedily on the table.",
        ]
    if method == "dqn":
        return [
            "A 128-128 neural network predicts Q(s, a) for the three torques",
            "directly from the raw 6-number observation.",
            f"Replay buffer of 100,000 transitions, mini-batch {params['batch_size']}, "
            f"target network synced every {params['target_sync_frequency']:,} steps.",
            "Acts greedily: picks the torque with the highest predicted Q.",
        ]
    return [
        "Actor network outputs a probability for each torque, pi(a | s);",
        "critic network estimates the state value V(s).",
        f"Clipped objective keeps each update small (clip = {params['clip_ratio']}), "
        f"rollouts of {params['rollout_steps']:,} steps, GAE lambda = 0.95.",
        "Demo plays the most likely action (deterministic policy).",
    ]


def title_card(method: str, result, summary: dict) -> Image.Image:
    image, draw = _canvas()
    accent = ACCENTS[method]
    draw.rectangle((0, 0, WIDTH, 8), fill=accent)
    _centered(draw, f"Acrobot-v1  ·  {NAMES[method]}", 110, H1)
    _centered(draw, "Swing the free end above the line using only the elbow motor", 170, BODY, MUTED)
    draw.rounded_rectangle((140, 240, WIDTH - 140, 450), radius=14, fill=PANEL)
    for row, line in enumerate(_description(method, result.spec["params"])):
        draw.text((180, 265 + row * 44), line, font=BODY, fill=TEXT)
    _centered(
        draw,
        f"Model shown: seed {result.seed}, the best of 5 seeds by final evaluation return "
        f"({result.final_mean:.1f}).",
        500, BODY, accent,
    )
    _centered(
        draw,
        f"Across all 5 seeds: return IQM {summary['iqm']:.1f}, success IQM {summary['success']:.0%}.",
        540, BODY, MUTED,
    )
    _centered(draw, "Next: three held-out episodes the agent never trained on  ·  6x real time", 620, SMALL, MUTED)
    return image


def learning_card(method: str, results: list, chosen_seed: int) -> Image.Image:
    accent = tuple(channel / 255 for channel in ACCENTS[method])
    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    figure.patch.set_facecolor(tuple(c / 255 for c in BACKGROUND))
    axis.set_facecolor(tuple(c / 255 for c in PANEL))
    steps = np.asarray(results[0].eval_steps)
    curves = np.asarray([result.eval_mean for result in results])
    for result, curve in zip(results, curves):
        chosen = result.seed == chosen_seed
        axis.plot(steps, curve, color=accent, alpha=1.0 if chosen else 0.3,
                  linewidth=2.5 if chosen else 1.2,
                  label=f"seed {result.seed} (shown in video)" if chosen else None)
    axis.plot(steps, [iqm(column) for column in curves.T], color="white", linewidth=2,
              linestyle="--", label="IQM across 5 seeds")
    axis.axhline(SOLVED_RETURN, color="#fbbf24", linewidth=1.2, linestyle=":", label="solved (-100)")
    axis.set_ylim(-520, -40)
    axis.set_xlabel("Environment interactions", color="white", fontsize=14)
    axis.set_ylabel("Held-out evaluation return", color="white", fontsize=14)
    axis.set_title(f"How {NAMES[method]} learned (10 held-out episodes every 5,000 interactions)",
                   color="white", fontsize=17, pad=14)
    axis.tick_params(colors="white", labelsize=12)
    for spine in axis.spines.values():
        spine.set_color("#394857")
    axis.grid(color="#394857", linewidth=0.6)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, facecolor="#1b2633", edgecolor="#394857", labelcolor="white", fontsize=12)
    figure.tight_layout(pad=2.0)
    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]).resize((WIDTH, HEIGHT))
    plt.close(figure)
    return image


def results_card(method: str, result, summary: dict, played: list[tuple[int, int, bool]]) -> Image.Image:
    image, draw = _canvas()
    accent = ACCENTS[method]
    draw.rectangle((0, 0, WIDTH, 8), fill=accent)
    _centered(draw, f"{NAMES[method]}: what the video showed", 70, H1)
    for row, (seed, length, success) in enumerate(played):
        y = 160 + row * 56
        draw.rounded_rectangle((260, y, WIDTH - 260, y + 44), radius=10, fill=PANEL)
        draw.text((290, y + 10), f"Held-out episode {seed:,}", font=BODY, fill=TEXT)
        outcome = f"goal in {length} steps" if success else f"time limit, {length} steps"
        draw.text((720, y + 10), outcome, font=BODY, fill=GOOD if success else BAD)
    _centered(draw, "Full study, 5 seeds x 100,000 interactions", 370, H2)
    columns = (("Return IQM", f"{summary['iqm']:.1f}"),
               ("95% CI", f"{summary['low']:.1f} to {summary['high']:.1f}"),
               ("Success IQM", f"{summary['success']:.0%}"),
               ("Steps to solve", summary["solve"]),
               ("Train time", f"{summary['seconds']:.1f} s"))
    width = (WIDTH - 200) // len(columns)
    for index, (label, value) in enumerate(columns):
        x = 100 + index * width
        draw.rounded_rectangle((x + 8, 420, x + width - 8, 520), radius=12, fill=PANEL)
        label_width = draw.textlength(label, font=SMALL)
        value_width = draw.textlength(value, font=H2)
        draw.text((x + (width - label_width) / 2, 438), label, font=SMALL, fill=MUTED)
        draw.text((x + (width - value_width) / 2, 470), value, font=H2, fill=accent)
    _centered(draw, "Return is about minus the number of steps, so closer to zero is better.", 570, BODY, MUTED)
    _centered(draw, "Results are conditional on this implementation, search space, seeds and budget.", 610, SMALL, MUTED)
    return image


# ---------------------------------------------------------------- episode frames

def _bars(draw, x: int, y: int, width: int, method: str, info: dict, action: int) -> None:
    values = info["values"]
    if method == "ppo":
        title, low, high, fmt = "Policy  pi(a | s)", 0.0, 1.0, "{:.0%}"
    else:
        title = "Q-table entries  Q(s, a)" if method == "q_learning" else "Network output  Q(s, a)"
        low, high, fmt = min(-100.0, float(values.min()) - 1.0), 0.0, "{:.2f}"
    draw.text((x, y), title, font=H2, fill=TEXT)
    for index, value in enumerate(values):
        row_y = y + 46 + index * 50
        chosen = index == action
        draw.text((x, row_y + 6), f"torque {TORQUES[index]:>2}", font=MONO, fill=TEXT if chosen else MUTED)
        bar_x, bar_w = x + 130, width - 240
        draw.rounded_rectangle((bar_x, row_y, bar_x + bar_w, row_y + 34), radius=6, fill=(38, 52, 68))
        fraction = 0.0 if high == low else (value - low) / (high - low)
        fill_w = int(bar_w * float(np.clip(fraction, 0.0, 1.0)))
        if fill_w > 0:
            colour = ACCENTS[method] if chosen else (84, 100, 118)
            draw.rounded_rectangle((bar_x, row_y, bar_x + fill_w, row_y + 34), radius=6, fill=colour)
        draw.text((bar_x + bar_w + 14, row_y + 6), fmt.format(value), font=MONO, fill=TEXT if chosen else MUTED)
    note_y = y + 46 + 3 * 50 + 4
    if method == "q_learning":
        cell = ",".join(str(index) for index in info["cell"])
        if info["visited"]:
            draw.text((x, note_y), f"cell ({cell})  ·  learned during training", font=SMALL, fill=MUTED)
        else:
            draw.text((x, note_y), f"cell ({cell})  ·  never updated: all zeros, default torque -1",
                      font=SMALL, fill=WARN)
    elif method == "dqn":
        draw.text((x, note_y), "picks the highest Q  (values ~ minus expected steps to go)", font=SMALL, fill=MUTED)
    else:
        draw.text((x, note_y), f"critic V(s) = {info['state_value']:.1f}  (expected return from here)",
                  font=SMALL, fill=MUTED)


def _height_plot(draw, x: int, y: int, width: int, height: int, heights: list[float]) -> None:
    draw.text((x, y), "Free-end height  h = -cos t1 - cos(t1 + t2)", font=SMALL, fill=MUTED)
    top, bottom = y + 26, y + height

    def to_y(value: float) -> float:
        return bottom - (value + 2.0) / 4.0 * (bottom - top)

    draw.rounded_rectangle((x, top, x + width, bottom), radius=8, fill=(22, 32, 44))
    draw.line((x, to_y(1.0), x + width, to_y(1.0)), fill=WARN, width=2)
    draw.text((x + width - 92, to_y(1.0) - 22), "goal h > 1", font=SMALL, fill=WARN)
    if len(heights) > 1:
        span = max(len(heights), 100)
        points = [(x + i / (span - 1) * width, to_y(h)) for i, h in enumerate(heights)]
        draw.line(points, fill=TEXT, width=2)


def episode_frame(method: str, env_frame: np.ndarray, episode: int, seed: int, step: int, episode_return: float,
                  info: dict, action: int, heights: list[float], banner: tuple[str, tuple] | None) -> Image.Image:
    image, draw = _canvas()
    accent = ACCENTS[method]
    draw.rectangle((0, 0, WIDTH, 6), fill=accent)
    draw.text((40, 22), f"{NAMES[method]}  ·  held-out episode {episode} of {len(EPISODE_SEEDS)}", font=H2, fill=TEXT)
    draw.text((40, 58), f"reset seed {seed:,}  ·  greedy policy  ·  6x real time", font=SMALL, fill=MUTED)

    size = 560
    frame = Image.fromarray(env_frame).resize((size, size), Image.Resampling.LANCZOS)
    draw.rounded_rectangle((36, 96, 40 + size + 4, 100 + size + 4), radius=10, fill=accent)
    image.paste(frame, (40, 100))
    if banner:
        text, colour = banner
        text_width = draw.textlength(text, font=H2)
        left = 40 + (size - text_width) / 2 - 20
        draw.rounded_rectangle((left, 590, left + text_width + 40, 640), radius=12, fill=(10, 14, 20))
        draw.text((left + 20, 600), text, font=H2, fill=colour)

    panel_x, panel_w = 640, 600
    draw.rounded_rectangle((panel_x, 100, panel_x + panel_w, 660), radius=14, fill=PANEL)
    inner = panel_x + 28
    stats = (("Step", f"{step}"), ("Return", f"{episode_return:.0f}"), ("Torque", TORQUES[action]))
    for index, (label, value) in enumerate(stats):
        sx = inner + index * 185
        draw.text((sx, 120), label, font=SMALL, fill=MUTED)
        draw.text((sx, 142), value, font=H1, fill=accent if label == "Torque" else TEXT)
    _bars(draw, inner, 215, panel_w - 56, method, info, action)
    _height_plot(draw, inner, 470, panel_w - 56, 170, heights)
    draw.text((40, 676), "Reward is -1 per step and 0 on the step that reaches the goal. Episodes stop at 500 steps.",
              font=SMALL, fill=MUTED)
    return image


def play_episodes(method: str, controller: Controller, writer) -> list[tuple[int, int, bool]]:
    env = gym.make("Acrobot-v1", max_episode_steps=500, render_mode="rgb_array")
    played = []
    for number, seed in enumerate(EPISODE_SEEDS, start=1):
        observation, _ = env.reset(seed=seed)
        heights = [tip_height(observation)]
        step = 0
        episode_return = 0.0
        terminated = truncated = False
        action, info = controller.decide(observation)
        writer.append_data(np.asarray(episode_frame(method, env.render(), number, seed, step, episode_return, info, action, heights, None)))
        while not (terminated or truncated):
            observation, reward, terminated, truncated, _ = env.step(action)
            step += 1
            episode_return += float(reward)
            heights.append(tip_height(observation))
            if not (terminated or truncated):
                action, info = controller.decide(observation)
            frame = episode_frame(method, env.render(), number, seed, step, episode_return, info, action, heights, None)
            writer.append_data(np.asarray(frame))
        banner = (f"Goal reached in {step} steps", GOOD) if terminated else (f"Time limit: {step} steps", BAD)
        final = np.asarray(episode_frame(method, env.render(), number, seed, step, episode_return, info, action, heights, banner))
        for _ in range(int(FPS * 1.8)):
            writer.append_data(final)
        played.append((seed, step, bool(terminated)))
    env.close()
    return played


# ---------------------------------------------------------------- main

def load_summary() -> dict[str, dict]:
    summary = {}
    with (RESULTS_ROOT / "tables" / "main_summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            steps = row["median_steps_to_solve"]
            summary[row["algorithm"]] = {
                "iqm": float(row["final_return_iqm"]), "low": float(row["return_ci_low"]),
                "high": float(row["return_ci_high"]), "success": float(row["success_rate_iqm"]),
                "solve": f"{int(float(steps)):,}" if steps else "not solved",
                "seconds": float(row["mean_training_seconds"]),
            }
    return summary


def hold(writer, image: Image.Image, seconds: float) -> None:
    frame = np.asarray(image)
    for _ in range(int(FPS * seconds)):
        writer.append_data(frame)


def render(method: str, results: list, summary: dict) -> Path:
    best = max(results, key=lambda result: result.final_mean)
    controller = Controller(best)
    path = VIDEO_DIR / f"{method}_walkthrough.mp4"
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=None) as writer:
        hold(writer, title_card(method, best, summary), 6)
        hold(writer, learning_card(method, results, best.seed), 6)
        played = play_episodes(method, controller, writer)
        hold(writer, results_card(method, best, summary, played), 7)
    print(f"Wrote {path}  (seed {best.seed}; episodes: {played})")
    return path


def main() -> None:
    summary = load_summary()
    main_results = load_group("main", RESULTS_ROOT)
    for method in ("q_learning", "dqn", "ppo"):
        results = sorted((r for r in main_results if r.method == method), key=lambda r: r.seed)
        render(method, results, summary[SUMMARY_NAMES[method]])


if __name__ == "__main__":
    main()
