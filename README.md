# Binoxxo Solver

[Binoxxo](https://en.wikipedia.org/wiki/Takuzu), aka Takuzu or Binairo, is a simple logic game.

The game consists of a 10x10 grid with some circles and crosses placed in it. The goal is to fill the empty cells with circles and crosses such that:
- No more than 2 similar symbols are aligned (`XOOXX` is ok but `OXXXO` is not)
- Each column and each row contains exactly 5 circles and 5 crosses
- All columns and all rows are different

This solver uses the webcam to detect a grid and try to solve it. The result is then display in augmented reality on the live camera feed.

## Requirements
- Python 3 or higher
- Python modules: `numpy`
- OpenCV for python: check [OpenCV's installation page](https://docs.opencv.org/4.x/da/df6/tutorial_py_table_of_contents_setup.html) for more info