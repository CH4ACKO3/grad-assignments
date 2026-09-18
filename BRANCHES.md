# Course branch layout

Organized on 2026-09-18. `main` contains shared files and navigation only.
Each named course branch contains its own course directory plus shared files.
Existing commit history is retained: historical commits may still contain
multiple course directories. This operation does not purge history.

## Preserved sources

The complete tree of each course directory is identical to the source below.

| Branch | Source commit | Course tree |
| --- | --- | --- |
| `csc6001` | `1ce9c4de2bd7262398d896ad9d8fd9adaec9f074` | `95d34122c9517551766cdfdedba07ddbaf542cfd` |
| `csc6022` | `1ce9c4de2bd7262398d896ad9d8fd9adaec9f074` | `a18d5a7667f1e17678ab2ecc31afeeac1b5a0bc7` |
| `csc6052` | `ee54404bae4bad7e31f53acdc8d07804386757f7` | `f61671c1aa1e2350c1aefe79941ddf5358744bc1` |
| `csc6124` | `b2ab54c73f6792ff5ed6c5673d6f3460d4a373dc` | `4ebe571ba4aca91c86450b9e8101c5b1ba718b9b` |
| `csc6129` | `c139036cd9898ee7fa0395753191afc161d680b9` | `1a49df94f67d6ab92a8329d566f3f9b14363560a` |
| `csc6300` | `9ef041ad569e1857a0b3fa900b972f68556ecd47` | `c9691381e597834d0b7de9dca61220c69ed2b1b3` |
| `dda6040` | `1ce9c4de2bd7262398d896ad9d8fd9adaec9f074` | `ffb12bb485b6536a2ed6c5023957481215ec9cdb` |

New organization commits extend the existing branch tips; no branch is force-pushed.
New branches CSC6001, CSC6022, and DDA6040 start from the previous main tip.
All assignment contents, PDFs, prompts, and course-local instructions are preserved.
Branches are long-lived course workspaces; changes to one course stay on its branch.
