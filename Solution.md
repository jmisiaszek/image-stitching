# RC HW 1

All tasks have their implementation within commented funtions. On the bottom of the code there are calls for functions for each subtask. To run them, you just uncomment them and run.

## Task 1

Solution to task 1 consists of map `marker_map` and functions `reprojection_error(...)`, `calibrate_six()`, `calibrate_one()` and `calibrate_and_compare()`. It calibrates camera using two required methods, calculates the error and outputs both values.

Results show that calibration reusing six times the same image has less error (around 0.06) than the method that includes the relative placement of tags (0.28 error). A possible cause to that is that we use same dataset many times, introducing redundancy. This lowers the reprojection error, but it might also lead to overfitting.

## Task 2

Solution to task 2 is in `apply_proj(...)` and `test_proj()`.

## Task 3

Solution to task 3 is in `find_transform(...)` and `test_transform()`.

## Task 4

Solution to task 4 is in `manual_transform()`.

Later tasks also use functions `project_canvas(...)`, `color_correction()`, `error_matrix()`, `optimal_seam()`, `blend_images()` and `crop()`.

## Task 5

Solution to task 5 is in `run_project_canvas()`.

## Task 6

Solution to task 6 is in `stitch_two_sg()`. Needed point pairs are in `./matching/`.

## Task 7

Solution to task 7 is in `stitch_five()`.