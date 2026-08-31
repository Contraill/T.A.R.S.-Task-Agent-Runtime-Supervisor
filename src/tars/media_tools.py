from __future__ import annotations

import hashlib
import importlib.util

from .artifact_tools import ArtifactRuntime


class ImageTools:
    """Deterministic raster operations backed by Pillow when installed."""

    def __init__(self, roots, *, runtime=None):
        self.artifacts = ArtifactRuntime(roots, runtime=runtime)

    @staticmethod
    def capabilities():
        available = bool(importlib.util.find_spec("PIL"))
        return {name: available for name in
                ("info", "resize", "crop", "rotate", "convert", "compress")}

    @staticmethod
    def _image():
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow image backend is unavailable") from exc
        return Image

    def info(self, path, *, task_id=None, session_id=None):
        actions, reads, _ = self.artifacts.authorize("image.info", (path,),
                                                     task_id=task_id, session_id=session_id)
        target = reads[0]
        try:
            Image = self._image()
        except RuntimeError as exc:
            return self.artifacts.result("image.info", actions,
                                         {"path": str(target), "available": False},
                                         state="unavailable", error=str(exc))
        with Image.open(target) as image:
            data = {"path": str(target), "format": image.format, "mode": image.mode,
                    "width": image.width, "height": image.height,
                    "frames": getattr(image, "n_frames", 1),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        return self.artifacts.result("image.info", actions, data, task_id=task_id,
                                     evidence_source=target)

    def transform(self, operation, source, output, *, size=None, box=None, angle=None,
                  format=None, quality=85, approval_ids=None, task_id=None, session_id=None):
        if operation not in {"resize", "crop", "rotate", "convert", "compress"}:
            raise ValueError(f"unsupported image operation: {operation}")
        actions, reads, writes = self.artifacts.authorize(
            f"image.{operation}", (source,), (output,), approval_ids=approval_ids,
            task_id=task_id, session_id=session_id,
            arguments={"operation": operation, "size": size, "box": box, "angle": angle,
                       "format": format, "quality": quality},
        )
        try:
            Image = self._image()
        except RuntimeError as exc:
            return self.artifacts.result(f"image.{operation}", actions,
                                         {"available": False}, state="unavailable", error=str(exc))
        source_path, destination = reads[0], writes[0]
        try:
            with Image.open(source_path) as image:
                transformed = image.copy()
                if operation == "resize":
                    if not size or min(size) <= 0:
                        raise ValueError("resize requires positive width and height")
                    transformed = transformed.resize(tuple(map(int, size)), Image.Resampling.LANCZOS)
                elif operation == "crop":
                    if not box or len(box) != 4:
                        raise ValueError("crop requires a four-value box")
                    transformed = transformed.crop(tuple(map(int, box)))
                elif operation == "rotate":
                    transformed = transformed.rotate(float(angle or 0), expand=True)
                destination.parent.mkdir(parents=True, exist_ok=True)
                save_format = format or destination.suffix.removeprefix(".") or image.format
                options = {"quality": max(1, min(100, int(quality))), "optimize": True}
                if save_format.upper() in {"PNG", "GIF", "BMP"}:
                    options.pop("quality")
                transformed.save(destination, format=save_format, **options)
            with Image.open(destination) as verified:
                dimensions = [verified.width, verified.height]
                verified_format = verified.format
            data = {"operation": operation, "source": str(source_path),
                    "output": str(destination), "format": verified_format,
                    "dimensions": dimensions, "quality": int(quality),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "bytes": destination.stat().st_size}
        except Exception as exc:
            self.artifacts.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        return self.artifacts.result(f"image.{operation}", actions, data, task_id=task_id,
                                     evidence_source=destination)
