import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from sftot_losses import ssim, unit_range


def _files(paths, domain):
    result = [Path(path).resolve() for path in paths]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{domain} must contain nonempty, unique file paths")
    if any(not path.is_file() for path in result):
        raise FileNotFoundError(f"{domain} contains a missing image")
    return result


def _index(paths, attribute, domain):
    result = {}
    for path in paths:
        key = getattr(path, attribute)
        if key in result:
            raise ValueError(f"Ambiguous {domain} {attribute}: {key!r}")
        result[key] = path
    return result


def _manifest_pairs(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if isinstance(data, dict):
        entries = list(data.items())
    elif isinstance(data, list):
        entries = []
        for row in data:
            if not isinstance(row, dict) or "input" not in row or "target" not in row:
                raise ValueError("Manifest rows must contain input and target basenames")
            entries.append((row["input"], row["target"]))
    else:
        raise ValueError("Manifest must be an input->target object or a list of pairs")
    result = {}
    targets = set()
    for source, target in entries:
        for name in (source, target):
            if (not isinstance(name, str) or not name or name in {".", ".."}
                    or "/" in name or "\\" in name or Path(name).name != name):
                raise ValueError("Manifest input/target values must be basenames")
        if source in result or target in targets:
            raise ValueError("Manifest must define a one-to-one mapping")
        result[source] = target
        targets.add(target)
    return result


def _check_transform(transform):


    if transform is None:
        return
    if not callable(transform):
        raise TypeError("Evaluation transform must be callable")
    name = type(transform).__name__
    if name.startswith("Random") or name in {"ColorJitter", "AutoAugment", "RandAugment", "AugMix"}:
        raise ValueError("Paired evaluation requires a deterministic image transform")
    for child in getattr(transform, "transforms", []):
        _check_transform(child)


def _pil_image(tensor):
    pixels = (tensor[0].clamp(0, 1).mul(255).round().byte()
              .permute(1, 2, 0).numpy())
    return Image.fromarray(pixels, mode="RGB")


class PairedMetrics:

    def __init__(self, dataset_folder, source_files, target_files, transform,
                 output_dir, manifest=None):
        sources = _files(source_files, "source_files")
        targets = _files(target_files, "target_files")
        _check_transform(transform)
        self.dataset_folder = str(Path(dataset_folder).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.transform = transform
        if manifest is None:
            source_index = _index(sources, "stem", "source")
            target_index = _index(targets, "stem", "target")
            if source_index.keys() != target_index.keys():
                raise ValueError("Paired evaluation requires identical filename stems; "
                                 "supply a manifest for different names")
            self.pairs = {source: target_index[stem] for stem, source in source_index.items()}
            self.pairing = {"method": "exact_stem", "manifest": None}
        else:
            source_index = _index(sources, "name", "source")
            target_index = _index(targets, "name", "target")
            mapping = _manifest_pairs(manifest)
            if (mapping.keys() != source_index.keys()
                    or set(mapping.values()) != target_index.keys()):
                raise ValueError("Manifest must cover exactly the supplied source and target files")
            self.pairs = {source_index[name]: target_index[target]
                          for name, target in mapping.items()}
            self.pairing = {"method": "manifest", "manifest": str(Path(manifest).resolve()),
                            "mapping": mapping}
        self.records = []
        self.seen = set()
        self.finished = False

    def add(self, source_path, input_tensor, prediction_tensor):

        source = Path(source_path).resolve()
        if self.finished or source in self.seen:
            raise ValueError("Cannot add after finalize or add the same source twice")
        if source not in self.pairs:
            raise ValueError(f"Source was not included in the validated pairing: {source}")
        if (input_tensor.shape != prediction_tensor.shape or input_tensor.ndim != 4
                or tuple(input_tensor.shape[:2]) != (1, 3)):
            raise ValueError("Paired metrics require equally shaped [1,3,H,W] input/prediction")
        raw_input = input_tensor.detach().float().cpu()
        raw_prediction = prediction_tensor.detach().float().cpu()
        if not torch.isfinite(raw_input).all() or not torch.isfinite(raw_prediction).all():
            raise ValueError("Paired metrics reject nonfinite input/prediction values")
        target_path = self.pairs[source]
        with Image.open(target_path) as image:
            target_image = image.convert("RGB")
            if self.transform is not None:
                target_image = self.transform(target_image)
            if not isinstance(target_image, Image.Image):
                raise TypeError("Evaluation transform must return a PIL image")
            pixels = np.array(target_image.convert("RGB"), dtype=np.float32, copy=True)
        target = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0) / 255.0
        if target.shape != raw_prediction.shape:
            raise ValueError(f"Transformed GT shape {tuple(target.shape)} differs from "
                             f"prediction {tuple(raw_prediction.shape)}: {target_path}")
        restored = unit_range(raw_prediction).clamp(0, 1)
        input_image = unit_range(raw_input).clamp(0, 1)

        def metrics(value):
            mse = (value - target).square().mean().item()
            return -10.0 * math.log10(max(mse, 1e-12)), ssim(value, target).item()

        psnr, similarity = metrics(restored)
        input_psnr, input_similarity = metrics(input_image)
        record = {
            "source_path": str(source), "target_path": str(target_path),
            "height": target.shape[-2], "width": target.shape[-1],
            "psnr_rgb": psnr, "ssim_rgb": similarity,
            "input_psnr_rgb": input_psnr, "input_ssim_rgb": input_similarity,
            "delta_psnr_rgb": psnr - input_psnr,
            "delta_ssim_rgb": similarity - input_similarity,
        }
        panels = [_pil_image(value) for value in (input_image, restored, target)]
        width, height = panels[0].size
        header = 22
        comparison = Image.new("RGB", (3 * width, height + header), "white")
        draw = ImageDraw.Draw(comparison)
        for index, (label, panel) in enumerate(zip(("Input", "Restored", "GT"), panels)):
            left = index * width
            draw.text((left + 4, 4), label, fill="black")
            comparison.paste(panel, (left, header))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"{len(self.records) + 1:05d}_{source.stem}_comparison.png"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite a comparison: {destination}")
        with destination.open("xb") as stream:
            comparison.save(stream, format="PNG")
        record["comparison_path"] = str(destination)
        self.records.append(record)
        self.seen.add(source)
        return record

    def finalize(self, step):

        if self.finished or not self.records:
            raise ValueError("Finalize requires new, nonempty metric records")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        names = ("psnr_rgb", "ssim_rgb", "input_psnr_rgb", "input_ssim_rgb",
                 "delta_psnr_rgb", "delta_ssim_rgb")
        mean = {name: sum(row[name] for row in self.records) / len(self.records) for name in names}
        report = {
            "step": step, "images": len(self.records), "expected_pairs": len(self.pairs),
            "all_pairs_evaluated": len(self.records) == len(self.pairs),
            "dataset_folder": self.dataset_folder, "pairing": self.pairing,
            "mean": mean, "per_image": self.records,
            "protocol": "RGB [0,1]; per-image mean; Gaussian valid SSIM; no border shave; PSNR MSE floor 1e-12",
            "transform": repr(self.transform),
            "used_for_training": False, "used_for_model_selection": False,
        }
        path = self.output_dir / "metrics.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite metrics: {path}")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
        summary = {key: value for key, value in report.items() if key != "per_image"}
        summary["metrics_path"] = str(path)
        with (self.output_dir.parent / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, allow_nan=False) + "\n")
        self.finished = True
        return report
