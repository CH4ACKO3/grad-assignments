# Graduate Assignments

This is the **CSC6001 — Analysis of Algorithms** branch. Its coursework is in `CSC6001-Analysis_of_Algorithms/`. Other courses live on their own branches.

## Courses

| Code | Course | Branch | Coursework |
| --- | --- | --- | --- |
| CSC6001 | Analysis of Algorithms | [`csc6001`](https://github.com/CH4ACKO3/grad-assignments/tree/csc6001) | [Files](https://github.com/CH4ACKO3/grad-assignments/tree/csc6001/CSC6001-Analysis_of_Algorithms) |
| CSC6022 | Machine Learning | [`csc6022`](https://github.com/CH4ACKO3/grad-assignments/tree/csc6022) | [Files](https://github.com/CH4ACKO3/grad-assignments/tree/csc6022/CSC6022-Machine_Learning) |
| CSC6052 | Natural Language Processing | [`csc6052`](https://github.com/CH4ACKO3/grad-assignments/tree/csc6052) | [Files](https://github.com/CH4ACKO3/grad-assignments/tree/csc6052/CSC6052-Natural_Language_Processing) |
| CSC6124 | Graph Computing | [`csc6124`](https://github.com/CH4ACKO3/grad-assignments/tree/csc6124) | [Files](https://github.com/CH4ACKO3/grad-assignments/tree/csc6124/CSC6124-Graph_Computing) |
| CSC6129 | Reinforcement Learning | [`csc6129`](https://github.com/CH4ACKO3/grad-assignments/tree/csc6129) | [Files](https://github.com/CH4ACKO3/grad-assignments/tree/csc6129/CSC6129-Reinforcement_Learning) |
| CSC6300 | Thesis Research (I) | [`csc6300`](https://github.com/CH4ACKO3/grad-assignments/tree/csc6300) | [Files](https://github.com/CH4ACKO3/grad-assignments/tree/csc6300/CSC6300-Thesis_Research_%28I%29) |
| DDA6040 | Dynamic Programming and Stochastic Control | [`dda6040`](https://github.com/CH4ACKO3/grad-assignments/tree/dda6040) | [Files](https://github.com/CH4ACKO3/grad-assignments/tree/dda6040/DDA6040-Dynamic_Programming_and_Stochastic_Control) |

CSC6124 currently contains the page 32 question and a blank answer template.

## Working with a course

Commit or stash your current work before switching branches.

```sh
git fetch origin
git switch csc6124
# On a fresh clone if the local branch does not exist:
# git switch --track origin/csc6124
```

Course folders keep their existing names and files. Compile LaTeX from inside
the course folder using its existing instructions. Python environments, when
needed, remain separate for each course.

`common_files/` holds shared information and `template/` holds reusable templates.
See [BRANCHES.md](BRANCHES.md) for the branch layout and migration record.

Commit assignment changes on the corresponding course branch. Do not merge a
whole course branch into main: main intentionally has no coursework folders.
For shared template changes, commit them separately on main and cherry-pick
that specific shared-files commit onto the course branches that need it.

## Statement on Code Usage, Templates, and Plagiarism

Feel free to reference and learn from the code, templates, and workflows in this repository. I encourage you to use them as a learning resource and inspiration.

### Permitted Usage:

Learning and Understanding: Study the code to grasp concepts and implementation details.
Template and Structure Reuse: Adapt and modify the project structure, file organization, code templates, or utility functions for your own projects.
Idea Generation: Draw inspiration for solving problems.

### Prohibited Usage: Plagiarism

Direct copying and submitting this code as your own original work, with minimal or no modifications, is strictly prohibited. This includes, but is not limited to, submitting the code for assignments or projects without proper attribution or significant original contribution.

### Disclaimer:

The code in this repository is provided for educational and learning purposes only. I am not responsible for any academic misconduct or consequences arising from the direct plagiarism of this code. By using this repository, you acknowledge and agree that you are solely responsible for your own academic integrity.


Please adhere to academic honesty principles. If you use this code as a reference, ensure you understand it, modify it significantly, and provide appropriate attribution as required by your institution.
