# Research paper build

The paper is written for LuaLaTeX. Install a TeX distribution containing
LuaLaTeX first; on macOS, BasicTeX is sufficient:

```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
```

Then, from the repository root, compile it with:

```bash
mkdir -p output/pdf
lualatex -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf paper/acrobot_rl_comparison.tex
BIBINPUTS="$PWD/paper:" bibtex output/pdf/acrobot_rl_comparison
lualatex -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf paper/acrobot_rl_comparison.tex
lualatex -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf paper/acrobot_rl_comparison.tex
```

The final document is `output/pdf/acrobot_rl_comparison.pdf`. Figures are loaded
from the experiment-generated vector PDFs in `results/study/figures/`.

Before university submission, update the author affiliation and replace the data
availability note with the final GitHub URL.
