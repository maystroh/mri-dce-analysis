"""Turn an untidy input folder into one standard folder per exam (`--organize`).

The input is never changed. Every exam is planned first; when one exam cannot be formed the
whole run stops before analysis (`OrganizeError`). Each planned exam then gets a folder of
links under `<out>/_organized/`, which the JSON exam route consumes. Results refer to the
original files, and each folder is removed once its exam is processed.

* DICOM files are found by the `DICM` marker, whatever their names, and grouped into exams by
  `StudyInstanceUID`. The exam folder is the deepest folder holding every file of the study.
* NIfTI phases are found by the existing name rules only; nothing is guessed. A reference
  (`Pre`) whose later phases (`Post_N`) sit in a sibling folder joins them at the nearest
  folder above that also holds them.
* When one exam folder holds both, the DICOM is used and the NIfTI ignored.

See docs/adr/0001-organize-exam-identity.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mri_analysis.discovery import _NON_PHASE, iter_exams

ORGANIZED_DIR = "_organized"
ORIGIN_FILE = ".origin.json"
NOT_DICOM = (".nii", ".nii.gz", ".json", ".csv", ".tsv", ".txt", ".md", ".png", ".jpg", ".jpeg",
             ".pdf", ".xml", ".html", ".zip", ".gz", ".py", ".sh", ".log", ".xlsx", ".npy", ".npz")
SKIPPED_DIRS = {"predictions", "per_breast_halves", "labels", "masks", ORGANIZED_DIR}
PRE = re.compile(r"pre\.nii(\.gz)?", re.IGNORECASE)
POST = re.compile(r"post_(\d+)\.nii(\.gz)?", re.IGNORECASE)


class OrganizeError(ValueError):
    """At least one exam of the input folder cannot be organized."""


@dataclass
class PlannedExam:
    case: str                 # name of the exam: its folder below the input root
    input_type: str           # "dicom" or "nifti"
    folder: str               # the exam folder (absolute)
    files: list[str]          # DICOM: every file; NIfTI: reference first, then phases in order
    study_uid: str | None = None
    ignored: list[str] = field(default_factory=list)


@dataclass
class Plan:
    root: str
    exams: list[PlannedExam] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {"input": self.root, "status": "error" if self.errors else "ok",
                "exams": [asdict(exam) for exam in self.exams],
                "errors": self.errors, "ignored": self.ignored}


def is_dicom(path: str) -> bool:
    if path.lower().endswith(NOT_DICOM):
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(132)[128:132] == b"DICM"
    except OSError:
        return False


def _walk(root: str, skip: set[str]):
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIPPED_DIRS
                         and os.path.abspath(os.path.join(directory, d)) not in skip)
        yield directory, sorted(f for f in files if not f.startswith("."))


def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def _case(root: str, folder: str) -> str:
    relative = os.path.relpath(folder, root)
    return os.path.basename(root) if relative == "." else Path(relative).as_posix()


def _dicom_exams(root: str, files: list[str], plan: Plan) -> None:
    import pydicom
    studies: dict[str, list[str]] = {}
    patients: dict[str, set[str]] = {}
    for path in files:
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True,
                                     specific_tags=["StudyInstanceUID", "PatientID"])
        except Exception as exc:
            plan.errors.append(f"{path}: unreadable DICOM header ({type(exc).__name__}: {exc})")
            continue
        uid = str(getattr(header, "StudyInstanceUID", "") or "").strip()
        if not uid:
            plan.errors.append(f"{path}: DICOM file has no StudyInstanceUID")
            continue
        studies.setdefault(uid, []).append(path)
        patients.setdefault(uid, set()).add(str(getattr(header, "PatientID", "") or "").strip())
    by_folder: dict[str, list[str]] = {}
    for uid, paths in studies.items():
        if len(patients[uid]) > 1:
            plan.errors.append(f"Study {uid} has several PatientID values: "
                               f"{', '.join(sorted(repr(p) for p in patients[uid]))}")
            continue
        by_folder.setdefault(os.path.commonpath([os.path.dirname(p) for p in paths]), []).append(uid)
    for folder, uids in sorted(by_folder.items()):
        for uid in sorted(uids):
            case = _case(root, folder)
            if len(uids) > 1:
                case += "_" + hashlib.sha256(uid.encode()).hexdigest()[:6]
            plan.exams.append(PlannedExam(case, "dicom", folder, sorted(studies[uid]), uid))


def _nifti_exams(root: str, folders: dict[str, list[str]], plan: Plan, dicom_folders: list[str]) -> None:
    pres, posts = [], {}
    for folder, names in sorted(folders.items()):
        named = [n for n in names if PRE.fullmatch(n) or POST.fullmatch(n)]
        if not named:
            exams = [e for e in iter_exams(folder, "auto", recursive=False)
                     if os.path.abspath(e.directory) == folder]
            if not exams:
                plan.errors.append(f"{folder}: NIfTI files whose phases cannot be identified by name "
                                   "(use Pre.nii.gz, Post_1.nii.gz, ... or {case}_0000.nii.gz, ...)")
            for exam in exams:
                case = _case(root, folder)
                if len(exams) > 1:
                    case = f"{case}/{exam.case}"
                plan.exams.append(PlannedExam(case, "nifti", folder, exam.phases))
            continue
        for name in named:
            path = os.path.join(folder, name)
            if PRE.fullmatch(name):
                pres.append(path)
            else:
                posts[path] = int(POST.fullmatch(name).group(1))
    claimed: set[str] = set()
    waiting = []
    # A reference with later phases beside it is complete where it is.
    for pre in pres:
        folder = os.path.dirname(pre)
        beside = [p for p in posts if os.path.dirname(p) == folder]
        if beside:
            claimed.update(beside)
            _add_nifti(root, folder, pre, beside, posts, plan)
        else:
            waiting.append(pre)
    # Otherwise its phases may sit in a sibling folder: join them one folder up, never higher.
    for pre in sorted(waiting):
        folder = os.path.dirname(pre)
        parent = os.path.dirname(folder) if folder != root else folder
        # Only Post files beside it in the parent, or in a sibling folder, can join it.
        below = [p for p in posts if p not in claimed and parent in (os.path.dirname(p),
                                                                      os.path.dirname(os.path.dirname(p)))]
        if not below:
            _add_nifti(root, folder, pre, [], posts, plan)
            continue
        rivals = ([p for p in pres if p != pre and _inside(p, parent)]
                  + [d for d in dicom_folders if _inside(d, parent)])
        if rivals:
            plan.errors.append(f"{parent}: cannot tell which exam owns the Post files beside {pre}; "
                               f"this folder also holds {', '.join(rivals)}. Put each exam in its own folder")
            claimed.update(below)
            continue
        claimed.update(below)
        _add_nifti(root, parent, pre, below, posts, plan)
    for post in sorted(set(posts) - claimed):
        plan.errors.append(f"{post}: Post file with no Pre in its folder or a sibling folder")


def _add_nifti(root, folder, pre, phases, numbers, plan) -> None:
    order = sorted(phases, key=lambda p: numbers[p])
    seen: dict[int, str] = {}
    for path in order:
        if numbers[path] in seen:
            plan.errors.append(f"{folder}: Post_{numbers[path]} appears twice "
                               f"({seen[numbers[path]]}, {path})")
            return
        seen[numbers[path]] = path
    plan.exams.append(PlannedExam(_case(root, folder), "nifti", folder, [pre, *order]))


def plan_input(root: str | Path, skip: list[str | Path] = ()) -> Plan:
    """Plan the exams of an input folder without changing anything."""
    root = os.path.abspath(root)
    plan = Plan(root)
    if not os.path.isdir(root):
        plan.errors.append(f"--organize needs a folder: {root}")
        return plan
    skipped = {os.path.abspath(s) for s in skip}
    dicom_files, nifti_folders = [], {}
    for directory, files in _walk(root, skipped):
        for name in files:
            if name.lower().endswith((".nii", ".nii.gz")):
                if not _NON_PHASE.search(name):
                    nifti_folders.setdefault(os.path.abspath(directory), []).append(name)
            elif is_dicom(os.path.join(directory, name)):
                dicom_files.append(os.path.abspath(os.path.join(directory, name)))
    if dicom_files:
        _dicom_exams(root, dicom_files, plan)
    dicom_folders = [e.folder for e in plan.exams]
    # DICOM wins: drop NIfTI found inside, above or at a DICOM exam folder.
    kept = {}
    for folder, names in nifti_folders.items():
        clash = [d for d in dicom_folders if _inside(folder, d) or _inside(d, folder)]
        if clash:
            plan.ignored.extend(os.path.join(folder, n) for n in names)
        else:
            kept[folder] = names
    _nifti_exams(root, kept, plan, dicom_folders)
    cases = [e.case for e in plan.exams]
    for case in sorted({c for c in cases if cases.count(c) > 1}):
        plan.errors.append(f"Two exams would share the name {case!r}")
    if not plan.exams and not plan.errors:
        plan.errors.append(f"No DICOM or NIfTI exam found in {root}")
    plan.exams.sort(key=lambda e: e.case)
    return plan


def _links_work(folder: Path) -> bool:
    target, link = folder / ".link_target", folder / ".link_test"
    try:
        target.write_text("")
        os.symlink(target, link)
        return link.resolve() == target.resolve()
    except OSError:
        return False
    finally:
        for path in (link, target):
            try:
                path.unlink()
            except OSError:
                pass


def build(plan: Plan, out: str | Path, name: str) -> Path:
    """Make one link folder per exam and the JSON exam list that points at them."""
    if plan.errors:
        raise OrganizeError("; ".join(plan.errors))
    out = Path(out).resolve()
    staging = out / ORGANIZED_DIR
    shutil.rmtree(staging, ignore_errors=True)   # leftovers of an interrupted run
    staging.mkdir(parents=True)
    links = _links_work(staging)
    mapping = {}
    for exam in plan.exams:
        folder = staging / re.sub(r"[^A-Za-z0-9._-]", "_", exam.case.replace("/", "__")).strip(".")
        folder.mkdir()
        if exam.input_type == "dicom":
            targets = [(f"{i:06d}.dcm", path) for i, path in enumerate(exam.files)]
        else:
            suffix = lambda p: ".nii.gz" if p.lower().endswith(".nii.gz") else ".nii"
            targets = [("Pre" + suffix(exam.files[0]), exam.files[0])]
            targets += [(f"Post_{i}{suffix(p)}", p) for i, p in enumerate(exam.files[1:], 1)]
            (folder / ORIGIN_FILE).write_text(json.dumps(dict(targets), indent=2) + "\n")
        for link_name, source in targets:
            if links:
                os.symlink(source, folder / link_name)
            else:
                shutil.copy2(source, folder / link_name)
        mapping[str(folder)] = {"organized_from": exam.folder, "organized_case": exam.case}
    exams_json = out / f"{name}_organized_exams.json"
    exams_json.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
    return exams_json


def origin(folder: Path) -> dict[str, str]:
    """Original file of each phase in an organized NIfTI folder ({} for any other folder)."""
    path = folder / ORIGIN_FILE
    return json.loads(path.read_text()) if path.is_file() else {}


def release(folder: str | Path, staging: str | Path) -> None:
    """Remove one exam's organized folder once it is processed."""
    folder, staging = Path(folder).resolve(), Path(staging).resolve()
    if folder.parent == staging:
        shutil.rmtree(folder, ignore_errors=True)


def describe(plan: Plan) -> str:
    lines = [f"Organize check of {plan.root}: {len(plan.exams)} exam(s), {len(plan.errors)} error(s)"]
    for exam in plan.exams:
        lines.append(f"  {exam.input_type:5}  {exam.case}  ({len(exam.files)} file(s))")
    if plan.ignored:
        lines.append(f"  ignored {len(plan.ignored)} NIfTI file(s) in DICOM exam folders")
    lines += [f"  ERROR  {error}" for error in plan.errors]
    return "\n".join(lines)
