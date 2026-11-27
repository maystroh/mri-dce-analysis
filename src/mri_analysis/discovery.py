"""Find the exams in a dataset, and the phases in an exam.

`mri_transforms.utils` already has three finders, and each one assumes a layout:
`find_nifti_files` (pre/post by name), `find_nifti_files_by_metadata` (the DICOM-derived
`{series}-{date}-{time}-{description}` convention), and `find_nifti_files_by_laterality`
(exams already split into `_left` / `_right`). None of them reads the two challenge
layouts this repository characterises, so the adapters live here:

* `mama_mia` — `{root}/{case}/{case}_0000.nii.gz`, `_0001`, ... (nnU-Net convention)
* `odelia`   — `{root}/{case}/Pre.nii.gz`, `Post_1.nii.gz`, ..., plus `Sub_*` and `T2`
* `auto`     — try each in turn, then fall back to `mri_transforms.find_nifti_files`

Phase order matters downstream: phase 0 is the Reference Phase every motion score is
measured against, so a finder that returns posts in the wrong order silently changes
`worst_phase`. Both adapters sort numerically, not lexically — `Post_10` after `Post_9`.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Not a phase of the DCE series: a subtraction is derived from two phases, and T2 is a
# different sequence. Both would corrupt a phase count and a motion score.
_NON_PHASE = re.compile(r"(^|[_-])(sub|t2|dwi|adc|seg|mask|label)", re.IGNORECASE)


@dataclass
class Exam:
    case: str
    directory: str
    pre: Optional[str]
    posts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    dicom_source: Optional[str] = None
    input_error: Optional[str] = None

    @property
    def phases(self) -> List[str]:
        return ([self.pre] if self.pre else []) + self.posts

    @property
    def n_phases(self) -> int:
        return len(self.phases)


def _niftis(directory: str) -> List[str]:
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(n for n in names if n.endswith((".nii", ".nii.gz")))


def _trailing_int(name: str) -> int:
    """The last integer in a filename, for numeric ordering. -1 when there is none."""
    found = re.findall(r"(\d+)", os.path.basename(name))
    return int(found[-1]) if found else -1


def find_mama_mia(directory: str, case: str) -> Optional[Exam]:
    """`{case}_0000.nii.gz` is the pre-contrast phase; higher indices are post-contrast."""
    phases = [n for n in _niftis(directory)
              if re.fullmatch(re.escape(case) + r"_\d{4}\.nii(?:\.gz)?", n)
              and not _NON_PHASE.search(n)]
    if not phases:
        return None
    phases.sort(key=_trailing_int)
    if _trailing_int(phases[0]) != 0:
        return None
    full = [os.path.join(directory, n) for n in phases]
    return Exam(case=case, directory=directory, pre=full[0], posts=full[1:])


def find_odelia(directory: str, case: str) -> Optional[Exam]:
    """`Pre.nii.gz` plus `Post_N.nii.gz`. `Sub_*` and `T2` are not phases."""
    names = _niftis(directory)
    pre = next((n for n in names if n.lower().startswith("pre")), None)
    posts = sorted((n for n in names
                    if n.lower().startswith("post") and not _NON_PHASE.search(n)),
                   key=_trailing_int)
    if pre is None and not posts:
        return None
    join = lambda n: os.path.join(directory, n)
    return Exam(case=case, directory=directory,
                pre=join(pre) if pre else None, posts=[join(n) for n in posts])


def find_generic(directory: str, case: str) -> Optional[Exam]:
    """Last resort: `mri_transforms.find_nifti_files`, which knows the naming the
    lesion pipeline uses. It asserts when it cannot classify, so failure is caught."""
    if not _niftis(directory):
        return None
    try:
        from mri_transforms.utils import find_nifti_files
        pre, posts = find_nifti_files(directory)
    except Exception:
        return None
    # The library finder walks recursively. Discovery must not collapse several patients
    # into one exam just because their parent directory was visited first.
    local = lambda p: (p and os.path.dirname(os.path.abspath(p)) == os.path.abspath(directory)
                       and not _NON_PHASE.search(os.path.basename(p)))
    pre = pre if local(pre) else None
    posts = [p for p in posts if local(p)]
    if pre is None:
        local_pres = [n for n in _niftis(directory)
                      if re.search(r"(^|[_-])pre([_.-]|$)", n, re.IGNORECASE)
                      and not _NON_PHASE.search(n)]
        if len(local_pres) == 1:
            pre = os.path.join(directory, local_pres[0])
    if not pre and not posts:
        candidates = [n for n in _niftis(directory) if not _NON_PHASE.search(n)]
        if len(candidates) == 1:
            return Exam(case, directory, os.path.join(directory, candidates[0]), [])
        return None
    return Exam(case=case, directory=directory, pre=pre, posts=list(posts))


LAYOUTS: Dict[str, Callable[[str, str], Optional[Exam]]] = {
    "mama_mia": find_mama_mia,
    "odelia": find_odelia,
    "generic": find_generic,
}


def find_exam(directory: str, case: str, layout: str = "auto") -> Optional[Exam]:
    if layout != "auto":
        return LAYOUTS[layout](directory, case)
    for finder in (find_mama_mia, find_odelia, find_generic):
        exam = finder(directory, case)
        if exam is not None and exam.phases:
            return exam
    return None


def iter_exams(root: str, layout: str = "auto", limit: Optional[int] = None,
               recursive: bool = False, exclude: Optional[List[str]] = None):
    """Discover exams without combining patients or counting empty folders toward the limit.

    Recursive case IDs retain the path below the dataset root so repeated study names in
    different collections cannot overwrite each other's predictions. Flat nnU-Net directories
    are grouped by the filename's case prefix. Ambiguous naming needs an explicit manifest.
    """
    root = os.path.abspath(root)
    excluded = {os.path.abspath(p) for p in (exclude or [])}
    found = 0
    for directory, dirs, _ in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                         and d not in ("predictions", "per_breast_halves", "labels", "masks")
                         and os.path.join(directory, d) not in excluded)
        relative = os.path.relpath(directory, root)
        names = _niftis(directory)
        prefixes = sorted({m.group(1) for n in names
                           if (m := re.fullmatch(r"(.+)_\d{4}\.nii(?:\.gz)?", n))
                           and not _NON_PHASE.search(n)})
        exams = []
        if prefixes and layout in ("auto", "mama_mia"):
            for prefix in prefixes:
                exam = find_mama_mia(directory, prefix)
                if exam:
                    exam.case = prefix if relative == "." else (
                        relative if prefix == os.path.basename(directory)
                        else os.path.join(relative, prefix))
                    exams.append(exam)
        if not exams:
            case = os.path.basename(root) if relative == "." else relative
            exam = find_exam(directory, os.path.basename(directory), layout)
            if exam:
                exam.case = case
                exams.append(exam)
        for exam in exams:
            yield exam
            found += 1
            if limit and found >= limit:
                return
        if not recursive and relative != ".":
            dirs[:] = []


def exams_from_manifest(path: str, limit: Optional[int] = None) -> List[Exam]:
    """Read any dataset through CSV: case, source_path, optional posts (ordered JSON array).

    Relative volume paths resolve beside the manifest. Extra columns are carried into the
    analysis as dataset metadata. Phase order is explicit; no arbitrary filename is treated
    as a pre-contrast phase when several unrecognised volumes exist.
    """
    base = Path(path).resolve().parent
    exams, seen = [], set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        source_column = "source_path" if "source_path" in columns else "image"
        if "case" not in columns or source_column not in columns:
            raise ValueError("manifest requires case and source_path (or image) columns")
        for line, row in enumerate(reader, 2):
            case = row["case"].strip()
            if (not case or Path(case).is_absolute() or ".." in Path(case).parts
                    or "\\" in case or case == "."):
                raise ValueError(f"manifest line {line}: unsafe or empty case ID {case!r}")
            if case in seen:
                raise ValueError(f"manifest line {line}: duplicate case {case!r}")
            seen.add(case)
            source = row[source_column].strip()
            if not source:
                raise ValueError(f"manifest line {line}: empty source_path")
            posts = json.loads(row.get("posts") or "[]")
            if not isinstance(posts, list) or not all(isinstance(p, str) for p in posts):
                raise ValueError(f"manifest line {line}: posts must be a JSON array of paths")
            def resolve_path(value):
                target = Path(value)
                return str(target if target.is_absolute() else base / target)
            pre = resolve_path(source)
            exams.append(Exam(case, os.path.dirname(pre), pre, [resolve_path(p) for p in posts],
                              {k: v for k, v in row.items()
                               if k not in ("case", "source_path", "posts") and k is not None}))
            if limit and len(exams) >= limit:
                break
    return exams
