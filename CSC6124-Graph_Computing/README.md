# CSC6124 — Graph Computing（图计算）

课程名称与代码取自 `../common_files/CS_Ph.D._Graduate_Handbook_2025.txt`。

## 本次任务

完成 lecture note **Topic 1: Graph Traversal-1，第 32 页 Quick exercise**。
老师的通知由用户提供，记录于 `prompt/quick_exercise_32.md`。
作业编号、截止时间、提交平台及文件名规则尚未提供。

- `quick_exercise_32.tex`：可编辑的提交模板，未填写答案。
- `tex/quick_exercise_32.pdf`：模板预览，当前不可作为已完成作业提交。
- `AGENTS.md`：本课程的辅助范围，供后续工具遵守。
- `code/data/`、`code/result/`：按仓库约定预留；本题不需要运行代码。

沿用仓库模板的 memoir 文档类、A4 纸张、11pt 字号和 3cm 左右页边距；
单题采用同页标题，不设独立封面或目录。

## 需要你自己完成

1. 打开你自己的课件第 32 页，阅读原图与题目中的代码，独立完成题目。
2. 将你自己的答案填入 `Answer`；如老师要求过程，在 `Working` 写入你自己的过程或说明，否则删除该节。模板未提供推导、答案或验证程序。
3. 核对姓名，填写 `StudentID` 和 `SubmissionDate`。目前姓名沿用仓库模板；学号没有从邮箱推断。
4. 核对课程通知中的作业编号、截止时间、提交平台、文件名及是否要求过程。不要把这里的建议格式当成老师的规定。
5. 删除所有方括号占位文字和不用的节，编译后打开 PDF 检查，再按老师要求命名并提交。

AI 的本次工作仅为仓库整理与空白模板排版。课程禁止 AI 解题；若课程另要求披露格式辅助，请按实际使用情况自行填写，模板不代写任何诚信声明。

## 编译

在本课程目录运行（需要已安装的 TeX Live / MiKTeX 与 latexmk）：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tex quick_exercise_32.tex
```

结果位于 `tex/quick_exercise_32.pdf`。本题无需 Python 环境或额外数据。
`pyproject.toml` 仅保留仓库的课程目录约定，不包含解题程序或依赖。
