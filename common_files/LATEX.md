# LaTeX layout and independent course worktrees

Each course branch contains exactly one course directory plus shared files.
`main` contains shared files only. Existing commit history is intentionally retained.

## Directory convention

```text
CourseCode-CourseTitle/
  .latexmkrc           Default compiler and output location
  *.tex                Main documents; preserve existing filenames
  prompt/              Assignment statements and supplied question assets
  code/
    data/              Input data
    result/            Experiment output and figures
  tex/                 Compiled PDFs and LaTeX temporary files
  pyproject.toml       Course-specific Python dependencies, when applicable
```

Existing `figs/`, code subdirectories, and auxiliary source paths are retained so
that historical documents continue to find their assets. Do not move a main .tex
file into `tex/`. Only `tex/<document>.pdf` is the current compiled PDF; avoid
duplicate generated PDFs beside the source. Git ignores auxiliary build files.

From the course directory:

```powershell
latexmk assignment_1.tex
```

Use the actual document filename. The course `.latexmkrc` sets PDF mode and
`tex/` output. CSC6011 and DDA6040 default to XeLaTeX for their Chinese-language templates;
other courses default to pdfLaTeX. Engine overrides remain available.
From the worktree root:

```powershell
.\common_files\build-latex.ps1 -Source .\CourseCode-CourseTitle\assignment_1.tex
```

For documents requiring external conversion (for example SVG/Inkscape), use
`-ShellEscape` with the helper; the converter must already be installed.
The helper also accepts `-Engine xelatex`, `pdflatex`, or `lualatex`.
Older minted documents may require a compatible minted/Pygments installation
and shell escape. This directory cleanup does not install dependencies or alter
the document's packages, code, figures, or answers.

Direct `pdflatex` calls do not read `.latexmkrc`; use latexmk or the helper.
For LaTeX Workshop set `latex-workshop.latex.outDir` to `%DIR%/tex` and pass
`-outdir=%OUTDIR%` and `-cd` in the latexmk recipe. Local workspace settings are
configured separately and remain ignored by Git.

## Worktrees

Local worktrees are managed with `git worktree list`. Open the matching folder
instead of switching a single folder between course branches. Branch switching
does not remove untracked files, Python environments, or ignored build files.
Keep local work in its course worktree; do not merge entire course branches
into main. Share isolated template/configuration commits as needed.
