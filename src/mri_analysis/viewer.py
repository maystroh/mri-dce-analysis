"""Browse saved analysis results and image overlays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import threading
import webbrowser
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from flask import Flask, abort, jsonify, request, send_file
from PIL import Image, ImageDraw
from scipy import ndimage

from mri_analysis.artifacts import discover_runs, file_path, read_csv, read_json

PLANES = {"sagittal": 0, "coronal": 1, "axial": 2}
COLOURS = {"prediction": (255, 181, 71), "ground_truth": (34, 211, 238),
           "overlap": (73, 214, 137)}
WEB = Path(__file__).with_name("web")


REVIEW_SCHEMA_VERSION = 1
REVIEW_DECISIONS = ("", "Accurate", "Incorrect", "Needs review")
REVIEW_MASKS = ("region", "nipple")
REVIEW_NOTE_LIMIT = 4000
REVIEW_EXPORT_LIMIT = 50 * 1024 * 1024


class ReviewStore:
    """Keep reviewer decisions in ``<run>_viewer_reviews.json`` and a CSV copy.

    One overall decision covers both masks. A per-mask decision, when set,
    replaces the overall decision for that mask only.
    """

    def __init__(self, root: Path, runs):
        self.root, self.runs = root, set(runs)
        self.lock = threading.Lock()

    def path(self, run: str) -> Path:
        return self.root / f"{run}_viewer_reviews.json"

    def load(self, run: str) -> dict:
        data = read_json(self.path(run))
        reviews = data.get("reviews") if isinstance(data, dict) else None
        return reviews if isinstance(reviews, dict) else {}

    @staticmethod
    def effective(review: dict) -> dict:
        return {
            mask: review.get(mask) or review.get("overall", "") for mask in REVIEW_MASKS
        }

    def save(self, run: str, case: str, payload: dict) -> dict:
        review = {}
        for key in ("overall", *REVIEW_MASKS):
            value = payload.get(key) or ""
            if value not in REVIEW_DECISIONS:
                raise ValueError(
                    f"{key} must be one of: {', '.join(d or 'empty' for d in REVIEW_DECISIONS)}"
                )
            review[key] = value
        note = payload.get("note") or ""
        if not isinstance(note, str) or len(note) > REVIEW_NOTE_LIMIT:
            raise ValueError(
                f"note must be text of at most {REVIEW_NOTE_LIMIT} characters"
            )
        review.update(
            note=note,
            effective=self.effective(review),
            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        with self.lock:
            reviews = self.load(run)
            if any(review[k] for k in ("overall", *REVIEW_MASKS)):
                reviews[case] = review
            else:
                reviews.pop(case, None)
            self._write(run, reviews)
        return review

    def _write(self, run: str, reviews: dict) -> None:
        document = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "run": run,
            "decisions": [d for d in REVIEW_DECISIONS if d],
            "rule": "A region or nipple decision replaces the overall decision for that mask; empty means same as overall.",
            "reviews": dict(sorted(reviews.items())),
        }
        rows = [
            {
                "case": case,
                "overall": r.get("overall", ""),
                "region": r.get("region", ""),
                "nipple": r.get("nipple", ""),
                "region_effective": r.get("effective", {}).get("region", ""),
                "nipple_effective": r.get("effective", {}).get("nipple", ""),
                "note": r.get("note", ""),
                "updated_at": r.get("updated_at", ""),
            }
            for case, r in document["reviews"].items()
        ]
        columns = [
            "case",
            "overall",
            "region",
            "nipple",
            "region_effective",
            "nipple_effective",
            "note",
            "updated_at",
        ]
        csv_text = io.StringIO()
        writer = csv.DictWriter(csv_text, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        for path, text in (
            (self.path(run), json.dumps(document, indent=2)),
            (self.root / f"{run}_viewer_reviews.csv", csv_text.getvalue()),
        ):
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)


def _true(value: Any) -> bool:
    return str(value).lower() in ("true", "1", "1.0")


def _point(image, value) -> list[int] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), value)
    if not np.isfinite(voxel).all():
        return None
    return [int(np.clip(round(v), 0, image.shape[i] - 1)) for i, v in enumerate(voxel)]


def _largest_center(mask: np.ndarray | None) -> list[int] | None:
    if mask is None or not mask.any():
        return None
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return [int(round(v)) for v in ndimage.center_of_mass(mask, labels, int(sizes.argmax()))]


class RunRegistry:
    """Join saved case records; cache only one exam's image data at a time."""

    def __init__(self, root: str, name: str | None = None):
        self.root = Path(root).resolve()
        self.runs = discover_runs(self.root)
        if name and name not in self.runs:
            raise ValueError(f"Run {name!r} not found; available: {', '.join(self.runs)}")
        self.initial_run = name or next(iter(self.runs))
        self.cases = {}
        for run, manifest in self.runs.items():
            exams = {r.get("case"): r for r in read_csv(self.root / f"{run}_exams.csv")}
            scores = {r.get("case"): r for r in read_csv(self.root / f"{run}_annotation_scores.csv")}
            motion = ({r["case"]: r for r in read_json(self.root / f"{run}_motion.json").get("cases", [])}
                      if manifest.get("motion_enabled", True) else {})
            sides = read_csv(self.root / f"{run}_stage2.csv")
            sides_by_case = {}
            for row in sides:
                sides_by_case.setdefault(row.get("case"), []).append(row)
            for item in manifest["cases"]:
                case = item["case"]
                side_rows = sides_by_case.get(case, [])
                metrics = exams.get(case, {})
                status = metrics.get("status", "available")
                needs_review = status not in ("ok", "available") or any(
                    _true(r.get("needs_review")) for r in side_rows)
                motion_record = motion.get(case, {"status": "not_requested", "reason": "Motion analysis was not requested"})
                needs_review = needs_review or bool(motion_record.get("needs_review"))
                self.cases[(run, case)] = {**item, "metrics": metrics, "scores": scores.get(case, {}),
                                           "sides": side_rows, "status": status,
                                           "needs_review": needs_review, "motion": motion_record,
                                           "dataset_metadata": item.get("metadata", {})}
        extensions = {".csv", ".json", ".png", ".jpg", ".svg", ".md", ".log", ".txt", ".pdf"}
        self.artifacts = [p for p in sorted(self.root.iterdir()) if p.is_file()
                          and p.suffix.lower() in extensions
                          and p.resolve().is_relative_to(self.root)]
        self.review_key = hashlib.sha256(json.dumps(self.runs, sort_keys=True).encode()).hexdigest()[:20]
        self.reviews = ReviewStore(self.root, self.runs)

    def case(self, run: str, case: str) -> dict:
        if (run, case) not in self.cases:
            raise KeyError("Unknown run or case")
        return self.cases[(run, case)]

    def overview(self) -> dict:
        runs = []
        for name, manifest in self.runs.items():
            cases = [{"case": item["case"], "status": item["status"],
                      "needs_review": item["needs_review"], "scores": item["scores"],
                      "motion": {key: item["motion"].get(key) for key in ("status", "score_mm")}}
                     for (run, _), item in self.cases.items() if run == name]
            cases.sort(key=lambda c: (not c["needs_review"], c["case"]))
            runs.append({"name": name, "portable": manifest.get("portable", False), "cases": cases,
                         "summary": read_json(self.root / f"{name}_stage2_summary.json"),
                         "geometry": read_json(self.root / f"{name}_summary.json"),
                         "annotations": read_json(self.root / f"{name}_annotation_summary.json"),
                         "motion": (read_json(self.root / f"{name}_motion_summary.json")
                                    if manifest.get("motion_enabled", True) else {})})
        return {"initial_run": self.initial_run, "runs": runs, "review_key": self.review_key,
                "artifacts": [{"id": i, "name": p.name, "size_bytes": p.stat().st_size,
                               "url": f"/api/artifact/{i}"} for i, p in enumerate(self.artifacts)]}

    @lru_cache(maxsize=1)
    def volumes(self, run: str, case: str) -> dict:
        record = self.case(run, case)
        source = file_path(self.root, record["files"].get("source"))
        if source is None or not source.is_file():
            raise ValueError("Source image is unavailable. Mount the original dataset, or open a portable bundle.")
        native = nib.load(source)
        if len(native.shape) not in (3, 4):
            raise ValueError("Viewer requires a 3D volume or a 4D volume with a reference frame")
        canonical = nib.as_closest_canonical(native)
        data = np.asarray(canonical.dataobj, dtype=np.float32)
        warnings = []
        if data.ndim == 4:
            data = data[..., 0]
            warnings.append("4D source: displaying frame 0. No motion analysis is inferred from it.")
        finite = data[np.isfinite(data)]
        sample = finite[::max(1, finite.size // 250_000)]
        low, high = np.percentile(sample, [1, 99.5]) if sample.size else (0, 1)
        high = max(float(high), float(low) + 1)
        result = {"image": canonical, "source": data, "warnings": warnings,
                  "window": [float(low), high]}
        landmark_path = file_path(self.root, record["files"].get("landmarks"))
        landmark_manifest = read_json(landmark_path) if landmark_path and landmark_path.is_file() else {}
        entries = [e for e in landmark_manifest.get("entries", [])
                   if e.get("source_file") == source.name]
        entry = entries[0] if len(entries) == 1 else {}
        prediction_valid = True
        if landmark_manifest:
            fingerprint = hashlib.sha1(np.ascontiguousarray(native.affine, dtype=np.float64).tobytes()).hexdigest()[:16]
            prediction_valid = (len(entries) == 1 and entry.get("source_shape") == list(native.shape)
                                and entry.get("source_affine_sha1") == fingerprint)
            if not prediction_valid:
                warnings.append("Stored prediction source identity differs; prediction overlays disabled.")
        for role in ("pred_region", "pred_nipple", "gt_region", "gt_nipple"):
            path = file_path(self.root, record["files"].get(role))
            if path is None:
                continue
            if role.startswith("pred_") and not prediction_valid:
                continue
            try:
                mask = nib.load(path)
                if mask.shape[:3] != native.shape[:3] or len(mask.shape) != 3 or not np.allclose(
                        mask.affine, native.affine, atol=1e-4, rtol=0):
                    raise ValueError("mask grid differs from the reference image")
                result[role] = np.asarray(nib.as_closest_canonical(mask).dataobj) > 0
            except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
                warnings.append(f"{role} unavailable: {exc}")
        selected = []
        for side in entry.get("sides", {}).values():
            for nipple in side.get("nipples", []):
                if nipple.get("biggest") and not nipple.get("filtered_out"):
                    mm = nipple.get("position_mm", {})
                    if all(k in mm for k in ("x", "y", "z")):
                        point = _point(canonical, [mm[k] for k in ("x", "y", "z")])
                        if point and prediction_valid:
                            selected.append(point)
        scored = _point(canonical, record["scores"].get("predicted_nipple_mm"))
        if scored and prediction_valid:
            selected = [scored]
        gt = (_point(canonical, record["scores"].get("annotated_nipple_mm"))
              or _largest_center(result.get("gt_nipple"))) if "gt_nipple" in result else None
        result["centres"] = {"prediction": selected, "ground_truth": [gt] if gt else []}
        result["landmarks"] = landmark_manifest
        return result

    def details(self, run: str, case: str) -> dict:
        record = self.case(run, case)
        public = {k: record[k] for k in ("case", "status", "needs_review", "metrics", "scores", "sides", "motion", "dataset_metadata")}
        public["dicom_source"] = record.get("dicom_source")
        try:
            # A saved score must not silently describe a changed reference image.
            identity = record["motion"].get("reference_identity")
            source = file_path(self.root, record["files"].get("source"))
            if identity and source and source.is_file():
                image = nib.load(source)
                fingerprint = hashlib.sha1(np.ascontiguousarray(image.affine, dtype=np.float64).tobytes()).hexdigest()[:16]
                if identity.get("source_shape") != list(image.shape) or identity.get("source_affine_sha1") != fingerprint:
                    public["motion"] = {**record["motion"], "status": "stale", "needs_review": True,
                                        "reason": "Saved motion reference grid differs from the displayed image"}
                    public["needs_review"] = True
            volumes = self.volumes(run, case)
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            return {**public, "available": False, "error": str(exc)}
        shape = list(volumes["source"].shape)
        centres = volumes["centres"]
        position = next(iter(centres["ground_truth"] or centres["prediction"]), [n // 2 for n in shape])
        errors = {}
        for structure in ("region", "nipple"):
            pred, gt = volumes.get(f"pred_{structure}"), volumes.get(f"gt_{structure}")
            if pred is not None and gt is not None:
                diff = pred != gt
                if diff.any():
                    errors[structure] = [int(diff.sum(axis=tuple(j for j in range(3) if j != i)).argmax())
                                         for i in range(3)]
        return {**public, "available": True, "shape": shape,
                "spacing": [float(x) for x in volumes["image"].header.get_zooms()[:3]],
                "position": position, "centres": centres, "error_slices": errors,
                "window": volumes["window"], "layers": [r for r in volumes if r.startswith(("gt_", "pred_"))],
                "warnings": volumes["warnings"], "landmarks": volumes["landmarks"]}


def _slice(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.flipud(np.take(data, index, axis=axis).T)


def render_slice(volumes: dict, plane: str, index: int, mode: str, layer: str,
                 opacity: float, window: tuple[float, float] | None = None) -> Image.Image:
    axis = PLANES[plane]
    if not 0 <= index < volumes["source"].shape[axis]:
        raise ValueError("Slice index outside volume")
    low, high = window or volumes["window"]
    image_slice = _slice(volumes["source"], axis, index)
    gray = (np.clip(np.nan_to_num((image_slice - low) / (high - low), nan=0), 0, 1) * 255).astype(np.uint8)
    image = np.repeat(gray[..., None], 3, axis=2)
    for structure in (("region", "nipple") if layer == "both" else (layer,)):
        def mask(role):
            volume = volumes.get(f"{role}_{structure}")
            return _slice(volume, axis, index) if volume is not None else np.zeros(gray.shape, dtype=bool)
        pred, gt = mask("pred"), mask("gt")
        overlays = []
        if mode == "prediction":
            overlays = [(pred, COLOURS["prediction"])]
        elif mode == "ground_truth":
            overlays = [(gt, COLOURS["ground_truth"])]
        elif mode == "errors":
            # Errors need annotations; an absent GT must not turn all predictions into errors.
            if f"gt_{structure}" in volumes and f"pred_{structure}" in volumes:
                overlays = [(pred & gt, COLOURS["overlap"]), (pred & ~gt, COLOURS["prediction"]),
                            (gt & ~pred, COLOURS["ground_truth"])]
        elif mode == "contours":
            overlays = [(pred ^ ndimage.binary_erosion(pred), COLOURS["prediction"]),
                        (gt ^ ndimage.binary_erosion(gt), COLOURS["ground_truth"])]
        for mask_value, colour in overlays:
            alpha = 1 if mode == "contours" else opacity
            image[mask_value] = (image[mask_value] * (1 - alpha) + np.asarray(colour) * alpha).astype(np.uint8)
    result = Image.fromarray(image)
    if layer in ("both", "nipple") and mode != "image":
        draw = ImageDraw.Draw(result)
        remaining = [j for j in range(3) if j != axis]
        for kind, points in volumes["centres"].items():
            if mode == "prediction" and kind == "ground_truth" or mode == "ground_truth" and kind == "prediction":
                continue
            for point in points:
                if abs(point[axis] - index) <= 1:
                    x, y = point[remaining[0]], gray.shape[0] - 1 - point[remaining[1]]
                    draw.line((x - 4, y, x + 4, y), fill=COLOURS[kind])
                    draw.line((x, y - 4, x, y + 4), fill=COLOURS[kind])
    # Preserve physical aspect ratio for anisotropic acquisitions.
    remaining = [j for j in range(3) if j != axis]
    spacing = volumes["image"].header.get_zooms()
    width, height = result.width * spacing[remaining[0]], result.height * spacing[remaining[1]]
    scale = 700 / max(width, height)
    return result.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.NEAREST)


def create_app(run_dir: str, name: str | None = None) -> Flask:
    registry = RunRegistry(run_dir, name)
    app = Flask(__name__, static_folder=str(WEB), static_url_path="/static")
    app.config["RUN_REGISTRY"] = registry

    @app.get("/")
    def index():
        return send_file(WEB / "index.html")

    @app.get("/api/run")
    def overview():
        return jsonify(registry.overview())

    @app.get("/api/case")
    def case():
        try:
            return jsonify(registry.details(request.args.get("run", ""), request.args.get("case", "")))
        except KeyError:
            abort(404)

    @app.get("/api/slice")
    def image_slice():
        try:
            plane, mode, layer = (request.args.get(k, default) for k, default in (
                ("plane", "sagittal"), ("mode", "contours"), ("layer", "both")))
            if plane not in PLANES or mode not in ("image", "contours", "errors", "prediction", "ground_truth") or layer not in ("both", "region", "nipple"):
                raise ValueError("Invalid plane, overlay mode or layer")
            opacity = float(request.args.get("opacity", "0.45"))
            if not np.isfinite(opacity) or not 0 <= opacity <= 1:
                raise ValueError("Opacity must be between 0 and 1")
            window = None
            if "low" in request.args and "high" in request.args:
                window = float(request.args["low"]), float(request.args["high"])
                if not np.isfinite(window).all() or window[1] <= window[0]:
                    raise ValueError("Window high must exceed low")
            volumes = registry.volumes(request.args.get("run", ""), request.args.get("case", ""))
            result = render_slice(volumes, plane, int(request.args.get("index", "0")), mode, layer, opacity, window)
            buffer = io.BytesIO()
            result.save(buffer, format="PNG")
            buffer.seek(0)
            return send_file(buffer, mimetype="image/png")
        except KeyError:
            abort(404)
        except (OSError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    def attachment(payload: dict, filename: str):
        response = jsonify(payload)
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response

    def safe_name(text: str) -> str:
        return "".join(c if c.isalnum() or c in "-." else "_" for c in text).strip("_") or "case"

    @app.get("/api/case/export")
    def case_export():
        run_name, case_id = request.args.get("run", ""), request.args.get("case", "")
        try:
            value = registry.details(run_name, case_id)
        except KeyError:
            abort(404)
        payload = {"run": run_name, "case": case_id, "status": value["status"],
                   "needs_review": value["needs_review"], "exam_qc": value["metrics"],
                   "breasts": value["sides"], "annotation_scores": value["scores"],
                   "motion": value["motion"], "landmarks": value.get("landmarks", {}),
                   "dataset_metadata": value["dataset_metadata"], "dicom_source": value.get("dicom_source"),
                   "warnings": value.get("warnings", []), "image_error": value.get("error")}
        return attachment(payload, f"{safe_name(run_name)}_{safe_name(case_id)}_qc.json")

    @app.get("/api/run/export")
    def run_export():
        name = request.args.get("run", "")
        run = next((r for r in registry.overview()["runs"] if r["name"] == name), None)
        if run is None:
            abort(404)
        payload = {"run": name, "landmarks": run["summary"], "geometry": run["geometry"],
                   "annotations": run["annotations"], "motion": run["motion"],
                   "cases": [{k: c[k] for k in ("case", "status", "needs_review")} for c in run["cases"]]}
        return attachment(payload, f"{safe_name(name)}_run_summary.json")

    @app.get("/api/reviews/download")
    def reviews_download():
        name, kind = request.args.get("run", ""), request.args.get("format", "csv")
        if name not in registry.runs or kind not in ("csv", "json"):
            abort(404)
        path = registry.reviews.path(name).with_suffix(f".{kind}")
        if not path.is_file():
            abort(404)
        return send_file(path, as_attachment=True)

    @app.post("/api/reviews/export")
    def reviews_export():
        # One zip: each run's saved review files as they are now on disk, plus
        # mask_reviews.json. The page builds mask_reviews.json, because it also holds
        # reviews kept only in this browser (read-only results folder).
        # JSON only: a cross-site form cannot post here without a CORS preflight.
        if (request.content_length or 0) > REVIEW_EXPORT_LIMIT:
            return jsonify(error="The export is too large"), 413
        payload = request.get_json(silent=True) if request.is_json else None
        summary = payload.get("mask_reviews") if isinstance(payload, dict) else None
        if not isinstance(summary, dict):
            return jsonify(error="Send mask_reviews as JSON"), 415
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("mask_reviews.json", json.dumps(summary, indent=2))
            with registry.reviews.lock:  # never a half-saved file
                for name in registry.runs:
                    for kind in ("csv", "json"):
                        path = registry.reviews.path(name).with_suffix(f".{kind}")
                        if path.is_file():
                            archive.write(path, path.name)
        buffer.seek(0)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return send_file(buffer, mimetype="application/zip", as_attachment=True,
                         download_name=f"{safe_name(registry.root.name)}_reviews_{stamp}.zip")

    @app.get("/api/reviews")
    def reviews():
        name = request.args.get("run", "")
        if name not in registry.runs:
            abort(404)
        return jsonify(run=name, reviews=registry.reviews.load(name),
                       file=registry.reviews.path(name).name,
                       writable=os.access(registry.root, os.W_OK))

    @app.post("/api/review")
    def save_review():
        # JSON only: a cross-site form cannot post here without a CORS preflight.
        payload = request.get_json(silent=True) if request.is_json else None
        if not isinstance(payload, dict):
            return jsonify(error="Send the review as JSON"), 415
        try:
            registry.case(payload.get("run", ""), payload.get("case", ""))
        except KeyError:
            abort(404)
        try:
            review = registry.reviews.save(payload["run"], payload["case"], payload)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except OSError as exc:
            return jsonify(error=f"Could not write the review file: {exc}"), 500
        return jsonify(review=review, file=registry.reviews.path(payload["run"]).name)

    @app.get("/api/artifact/<int:artifact_id>")
    def artifact(artifact_id):
        if not 0 <= artifact_id < len(registry.artifacts):
            abort(404)
        path = registry.artifacts[artifact_id]
        if request.args.get("table") == "1" and path.suffix == ".csv":
            return jsonify(rows=read_csv(path))
        return send_file(path, as_attachment=request.args.get("download") == "1")

    @app.after_request
    def local_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        return response

    return app


def serve(run_dir: str, name: str | None = None, host: str = "127.0.0.1", port: int = 8060,
          open_browser: bool = False) -> None:
    from waitress.server import create_server

    app = create_app(run_dir, name)
    server = create_server(app, host=host, port=port, threads=2)
    url = f"http://{host}:{server.effective_port}"
    print(f"Viewer ready: {url}\nRun folder: {Path(run_dir).resolve()}\nCtrl+C stops the viewer.", flush=True)
    if open_browser:
        timer = threading.Timer(1, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="analysis output folder or exported portable bundle")
    parser.add_argument("--name", help="select a run prefix when the folder contains several runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--open", action="store_true", help="open the local viewer in your browser")
    args = parser.parse_args()
    try:
        serve(args.run_dir, args.name, args.host, args.port, args.open)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0
