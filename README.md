Notes from the Splatting Team

`models` is downloaded from the original 3DGS repo README section where they claimed as "pre-trained models", but wouldn't work if I just run `render.py` on it.

`images` is also downloaded from the original 3DGS repo README section containing all corresponding to all their "pre-trained models".

`tandt_db` is downloaded from a colab tutorial [here](https://github.com/camenduru/gaussian-splatting-colab/blob/main/gaussian_splatting_colab.ipynb), and the images for playroom in it actually work

`playroom` is a model trained by Stanley using the image in `tandt_db/db/playroom`

command for training: `python train.py -s .\tandt_db\db\playroom -m playroom_skm_30000 --iterations 30000`

command for rendering: `python render_single_view.py -m playroom -s tandt_db\db\playroom --view_index 23 --skip_test`

I created a separate conda environment `guassian_splatting_original` which uses the `dr_aa` branch of their `diff-gaussian-rasterization` implementation.

Command to use our own version of `diff_gaussian_rasterization`: `pip uninstall diff_gaussian_rasterization ; cd submodules/diff-gaussian-rasterization ; pip install -e . ` 

------

For updating the submodule and applying it:

Assume you edited .gitmodules to switch a submodule to track a different branch, say main → stanley.

1. Save the .gitmodules edit and commit it:
```bash
git add .gitmodules
git commit -m "Changed submodule branch to stanley"
```
2. Sync config to .git/config:
```bash
git submodule sync
```
This tells Git to actually apply your change from .gitmodules to its internal tracking config.

3. Re-initialize (if needed) and checkout correct branch in submodule:
```bash
git submodule update --init --remote
```

This will fetch the latest commit from the new branch and update the submodule.

-----
```bash
git clean -xfd
```
-----

For changes in the submodule,

### 🔀 `modified: submodules/diff-gaussian-rasterization (new commits, modified content)`

This means:

* You've checked out a different commit or branch **inside the submodule**
* Or made code changes inside the submodule directory

To **commit the updated submodule reference** (i.e., the new commit SHA):

```bash
git add submodules/diff-gaussian-rasterization
```

If you want to discard submodule changes (reset to the last committed SHA):

```bash
cd submodules/diff-gaussian-rasterization
git restore .
git checkout <original-branch-or-commit>
cd ../..
git add submodules/diff-gaussian-rasterization
```

---

### ❓ `modified: submodules/simple-knn (untracked content)`

This means there are **new files** in that submodule that haven’t been staged or committed.

1. To **inspect**:

   ```bash
   cd submodules/simple-knn
   git status
   ```

2. If those files are important:

   * Commit them inside the submodule:

     ```bash
     git add .
     git commit -m "Your message"
     cd ../..
     git add submodules/simple-knn
     ```

3. If you want to discard:

   ```bash
   cd submodules/simple-knn
   git clean -fd  # deletes untracked files
   cd ../..
   ```

---

Let me know if you want to keep any of the submodule changes local or push them to a remote too.
