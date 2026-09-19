# CSC6124 — Graph Computing（图计算）

课程名称与代码取自 `../common_files/CS_Ph.D._Graduate_Handbook_2025.txt`。

## 本次任务

完成 lecture note **Topic 1: Graph Traversal-1，第 32 页 Quick exercise**。
老师的通知由用户提供，记录于 `prompt/quick_exercise_32.md`。
作业编号、截止时间、提交平台及文件名规则尚未提供。

- `quick_exercise_32.tex`：主文档，在这里编辑自己的答案。
- `prompt/page32_graph.tex`、`prompt/page32_pseudocode.txt`：第 32 页原题的图和伪代码，编译时需与主文件一起保留。
- `tex/quick_exercise_32.pdf`：由当前主文档编译生成的 PDF；是否可提交由你核对内容后决定。
- `AGENTS.md`：本课程的辅助范围，供后续工具遵守。
- `code/data/`、`code/result/`：按仓库约定预留；本题不需要运行代码。

沿用仓库模板的 memoir 文档类、A4 纸张、11pt 字号和 3cm 左右页边距；
单题采用紧凑的同页标题，不设独立封面或目录。题目图形由原课件形状坐标转录，伪代码仅统一缩进。

## 需要你自己完成

1. 打开你自己的课件第 32 页，阅读原图与题目中的代码，独立完成题目。
2. 将你自己的答案填入 `Answer`；如老师要求过程，在 `Working` 写入你自己的过程或说明，否则删除该节。模板未提供推导、答案或验证程序。
3. 核对姓名，填写 `StudentID` 和 `SubmissionDate`。目前姓名沿用仓库模板；学号没有从邮箱推断。
4. 核对课程通知中的作业编号、截止时间、提交平台、文件名及是否要求过程。不要把这里的建议格式当成老师的规定。
5. 删除所有方括号占位文字和不用的节，编译后打开 PDF 检查，再按老师要求命名并提交。

AI 的本次工作仅为仓库整理与空白模板排版。课程禁止 AI 解题；若课程另要求披露格式辅助，请按实际使用情况自行填写，模板不代写任何诚信声明。

## 文件结构与编译

沿用仓库约定：主 `.tex` 放课程根目录，`tex/` 是编译输出目录，不是源码目录。

```text
CSC6124-Graph_Computing/
  .latexmkrc                 默认编译配置
  quick_exercise_32.tex      主文档
  prompt/                   题目说明、图形源码与伪代码
  tex/
    quick_exercise_32.pdf    唯一的当前 PDF
    ...                     aux、log、synctex 等临时产物
```

保留 `prompt/` 与主文档的相对位置；不要把主 `.tex` 单独移到 `tex/`。

在本课程目录运行（需要已安装的 TeX Live / MiKTeX 与 latexmk）：

```powershell
latexmk quick_exercise_32.tex
```

`.latexmkrc` 会默认生成 PDF，并把 PDF 和编译临时文件统一写入 `tex/`。
从仓库根目录编译时使用：

```powershell
latexmk -cd -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -outdir=tex CSC6124-Graph_Computing/quick_exercise_32.tex
```

若使用 VS Code / Cursor 的 LaTeX Workshop，设置 `latex-workshop.latex.outDir`
为 `%DIR%/tex`，并让 latexmk recipe 使用 `-outdir=%OUTDIR%`。
当前本地工作区已配置；`.vscode/` 按仓库规则不提交。
直接运行 `pdflatex` 不会读取 `.latexmkrc`，因此建议统一使用上述 latexmk 命令。

结果位于 `tex/quick_exercise_32.pdf`，课程根目录不保留第二份 PDF。
编译临时文件由仓库 `.gitignore` 忽略。本题无需 Python 环境或额外数据。
`pyproject.toml` 仅保留仓库的课程目录约定，不包含解题程序或依赖。
