# CSC6011 — Theory of Computation（计算理论）

独立分支：`csc6011`。本课程目录采用仓库统一的 LaTeX 文件结构。

## Homework 1

- `homework1.tex`：可编辑的作业主文件，从老师模板原样复制，答案尚未填写。
- `prompt/homework1.pdf`：老师提供的题目 PDF，保留原文件。
- `prompt/homework1-template.tex`：老师提供的原始模板，保留原文件。
- `tex/homework1.pdf`：编译后的文档；初始版本是空白答案模板。
- `.latexmkrc`：XeLaTeX 配置，PDF 和临时文件统一写入 `tex/`。
- `code/data/`、`code/result/`：仓库约定的预留目录。

根据所提供的作业文件，本次为 Fall 2026 Homework 1，涵盖 Lectures 1–4，满分 100 分。
截止时间为 **2026 年 9 月 30 日 23:59（UTC+8）**，在 Blackboard 提交 PDF。
文件中的课程规则允许 AI 辅助，但要求先自行尝试、自己撰写答案，不允许直接复制 AI 输出提交。
这些信息是作业文件的内容记录，不表示本次工作包含解题或提交作业。

## 编辑与编译

在 `homework1.tex` 中填写姓名和自己的答案，保持 `prompt/` 中的原文件不变。
在本课程目录运行：

```powershell
latexmk homework1.tex
```

在 worktree 根目录也可运行：

```powershell
.\common_files\build-latex.ps1 -Source .\CSC6011-Theory_of_Computation\homework1.tex
```

使用 **XeLaTeX**，不要使用 pdfLaTeX。输出仅放在 `tex/`，不要将主文件搬进输出目录。
本地 LaTeX Workshop 已配置同样的编译器及输出路径；`.vscode/` 不纳入版本控制。
编译需要 TeX Live 或 MiKTeX、latexmk，以及 ctex/Fandol 字体支持。
