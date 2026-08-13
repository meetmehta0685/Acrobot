"""Generate the self-contained Acrobot Colab notebook from the standalone script."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "colab" / "acrobot_colab.py"
NOTEBOOK_PATH = ROOT / "colab" / "acrobot_colab.ipynb"


def main() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    library_source = source.split('\nif __name__ == "__main__":', maxsplit=1)[0]

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "colab": {"name": "acrobot_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Acrobot Reinforcement Learning Comparison

This notebook implements and compares **Q-Learning, DQN, and PPO from scratch**
on Gymnasium's `Acrobot-v1`. It does not use Stable-Baselines3 or another
ready-made reinforcement-learning algorithm.

Run the cells from top to bottom. For a faculty presentation, use the quick
configuration first, then show the repository's completed 111-run research study.
The notebook is the portable demo; `run_study.py` is the complete experiment
runner with tuning, held-out evaluation, sweeps, ablations, and bootstrap CIs."""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Problem and experimental design

- **Observation:** 6 continuous values describing joint-angle sine/cosine values and angular velocities.
- **Action:** one of 3 torques: negative, zero, or positive.
- **Reward:** -1 per step until the target height is reached.
- **Fairness:** every algorithm receives the same environment-interaction budget.
- **Evaluation:** exploration is disabled and the same evaluation protocol is used for all methods.
- **Reproducibility:** Python, NumPy, PyTorch, exploration, and Gymnasium are seeded."""
        ),
        nbf.v4.new_markdown_cell("## 2. From-scratch implementation"),
        nbf.v4.new_code_cell(library_source),
        nbf.v4.new_markdown_cell(
            """## 3. Choose experiment size

Use 2,000 interactions and one seed only to verify that everything runs. For the
final report, use at least 100,000 interactions and five seeds. A full experiment
can take a long time, so run it before your presentation."""
        ),
        nbf.v4.new_code_cell(
            """#@title Experiment configuration
BUDGET = 2000 #@param {type:"integer"}
SEEDS = [0] #@param
EVALUATION_EPISODES = 2 #@param {type:"integer"}
OUTPUT_DIRECTORY = Path('/content/acrobot_colab_results')

print(f'Budget: {BUDGET:,} interactions per algorithm and seed')
print(f'Seeds: {SEEDS}')
print(f'Device: {choose_device()}')"""
        ),
        nbf.v4.new_markdown_cell("## 4. Verify mathematical components"),
        nbf.v4.new_code_cell("smoke_test()"),
        nbf.v4.new_markdown_cell(
            """## 5. Train and compare

This cell trains all three agents, evaluates them without exploration, saves raw
CSV data and checkpoints, and generates a comparison plot. Short runs validate
the pipeline but are not valid evidence that one algorithm is better."""
        ),
        nbf.v4.new_code_cell(
            "run_comparison(BUDGET, SEEDS, EVALUATION_EPISODES, OUTPUT_DIRECTORY)"
        ),
        nbf.v4.new_markdown_cell("## 6. Inspect the raw evidence"),
        nbf.v4.new_code_cell(
            """summary = pd.read_csv(OUTPUT_DIRECTORY / 'summary.csv')
display(summary)
display(plt.imread(OUTPUT_DIRECTORY / 'comparison.png'))
print('Raw files:', sorted(path.name for path in OUTPUT_DIRECTORY.iterdir()))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Faculty demonstration checklist

1. Explain why Acrobot is underactuated.
2. Identify the six observations, three actions, and -1 step reward.
3. Show Q-Learning's discretizer and Bellman update.
4. Show DQN's replay buffer and target network.
5. Show PPO's actor, critic, GAE, and clipped objective.
6. Explain why equal interaction budgets are fairer than equal episode counts.
7. Present evaluation return, success rate, stability across seeds, and training time.
8. Open the raw CSV to prove that the plots use actual experiment data.
9. State limitations: discretization error, seed sensitivity, and finite training budget.

The completed local study found PPO to be the only reliable solver at 100,000
interactions: final held-out return IQM -81.7 and success IQM 100% across five
seeds. DQN was seed-sensitive and Q-Learning did not reliably solve within the
budget. Present this as a conditional result, not a universal algorithm ranking."""
        ),
    ]
    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Created {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
