# Review the results in the viewer

The viewer opens one exam at a time with its MRI and the predicted breast region and
nipple masks. It shows one view at a time: **Axial** by default; pick **Sagittal** or
**Coronal** in the **View** list.

To review the masks:

1. Pick **All masks**: `Accurate`, `Incorrect` or `Needs review` (keys `1`, `2`, `3`).
2. If one mask differs, change only **Breast region** or **Nipple**.
3. Add a note if you want.
4. Click **Save & next** (or press `Enter`). The viewer saves and opens the next exam
   that is not reviewed yet.

**‹ Previous** and **Next ›** move between exams without saving. In the case list, a
reviewed exam is green (both masks accurate), red (a mask is incorrect) or orange (a
mask needs review). **Not reviewed yet only** hides the done exams. **QC review queue
only** shows the exams that the analysis flagged itself (failed exam, unusual breast
region, missing nipple or motion flag).

Each **Save & next** updates `<name>_viewer_reviews.json` and `<name>_viewer_reviews.csv`
in the results folder on your computer, not inside the Docker image. The viewer loads
these files each time it opens, so you can stop and continue later.

**Export reviews** downloads one zip file, `<results folder>_reviews_<date-time>.zip`. It
holds `<name>_viewer_reviews.csv` and `<name>_viewer_reviews.json` of every run (dataset)
with saved reviews, as they are now, and `mask_reviews.json` with one line per exam:
breast region and nipple decision, note and time. Send this zip to share the reviews.

Below the image, the viewer shows motion between phases (runs made with `--motion`),
quality control tables for the exam and for the run, and **Downloads** for the exam QC,
the run summary, the reviews and every file in the results folder.

On a server, `mri-docker.sh viewer` is reachable from other computers at
`http://<server address>:8060` (the script prints the address). The viewer has no login:
use it only on a trusted network, or set `VIEWER_BIND=127.0.0.1` to keep it on the
server. See [Viewer on a server](DOCKER.md#viewer-on-a-server) and
[Reviewing and sharing results](VIEWER_PACKAGING.md).
