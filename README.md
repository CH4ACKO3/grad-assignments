# Graduate Assignments

This repository contains my personal course assignments' solutions and source code. 

## Course Overview

| Course | PDFs | Code | LaTeX Source |
|--------|------|------|-------|
| **CSC6001** - Analysis of Algorithms | [A1](CSC6001-Analysis_of_Algorithms/tex/assignment_1.pdf) <br> [A2](CSC6001-Analysis_of_Algorithms/tex/assignment_2.pdf) <br> [A3](CSC6001-Analysis_of_Algorithms/tex/assignment_3.pdf) | [Code](CSC6001-Analysis_of_Algorithms/code/) | [A1](CSC6001-Analysis_of_Algorithms/assignment_1.tex) <br> [A2](CSC6001-Analysis_of_Algorithms\assignment_2.tex) <br> [A3](CSC6001-Analysis_of_Algorithms\assignment_3.tex) |
| **CSC6022** - Machine Learning | [A1](CSC6022-Machine_Learning/prompt/Assignment1_new.pdf) <br> [A2](CSC6022-Machine_Learning/prompt/Assignment2.pdf) | [Code](CSC6022-Machine_Learning/code/) | [A1](CSC6022-Machine_Learning/tex/assignment_1.pdf) <br> [A2](CSC6022-Machine_Learning/tex/assignment_2.pdf) |
| **DDA6040** - Dynamic Programming and Stochastic Control | [A1](DDA6040-Dynamic_Programming_and_Stochastic_Control/prompt/DDA6040_assignment1.pdf) | - | [A1](DDA6040-Dynamic_Programming_and_Stochastic_Control/tex/assignment_1.pdf) |

## Repository Structure

- `common_files/` - Shared course information and utilities
- `template/` - Template for new course folders
- `CourseCode-CourseTitle/` - Individual course folders containing:
  - `prompt/` - Assignment PDF files
  - `code/` - Source code and data files
  - `tex/` - Compiled LaTeX documents
  - `pyproject.toml` - Python project configuration (uv/venv)
  - `assignment_*.tex` - LaTeX source files

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python environment management. Each course folder has its own `pyproject.toml` file defining the project dependencies.

### Setting up a project environment:

```bash
# Navigate to the course folder
cd CourseCode-CourseTitle

# Create and activate a virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### Using the environment:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run Python scripts
python code/your_script.py
```

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
