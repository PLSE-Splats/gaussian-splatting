Notes from the Splatting Team

`models` is downloaded from the original 3DGS repo README section where they claimed as "pre-trained models", but wouldn't work if I just run `render.py` on it.

`images` is also downloaded from the original 3DGS repo README section containing all corresponding to all their "pre-trained models".

`tandt_db` is downloaded from a colab tutorial [here](https://github.com/camenduru/gaussian-splatting-colab/blob/main/gaussian_splatting_colab.ipynb), and the images for playroom in it actually work

`playroom` is a model trained by Stanley using the image in `tandt_db/db/playroom`

command for training: `python train.py -s .\tandt_db\db\playroom -m playroom --iterations 30000`

command for rendering: `python render_single_view.py -m playroom -s tandt_db\db\playroom --view_index 23 --skip_test`