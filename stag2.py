#!/usr/bin/env python3
# STAG (VisionModel edition): VisionModel inference + STAG filesystem/XMP flow

import argparse
import os
import signal
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms.functional as TVF
from torch.amp import autocast
from PIL import Image, ImageOps, UnidentifiedImageError
import rawpy
from pillow_heif import register_heif_opener

# --- Your local modules ---
from Models import VisionModel
from xmphandler import XMPHandler

VERSION = "2.0.0-vm"

# -------------------- Formats --------------------
RAW_EXTS = {
    ".3fr",".ari",".arw",".bay",".cr2",".cr3",".cap",".data",".dcr",".dng",".drf",".eip",
    ".erf",".fff",".gpr",".iiq",".k25",".kdc",".mdc",".mef",".mos",".mrw",".nef",".nrw",
    ".orf",".pef",".ptx",".pxn",".r3d",".raf",".raw",".rwl",".rw2",".rwz",".sr2",".srf",
    ".srw",".x3f"
}
NON_RAW_EXTRA = {".heic", ".heif", ".avif"}  # Pillow-HEIF handles these

# -------------------- Helpers --------------------
def iter_files(root: Path):
    for dirpath, _, files in os.walk(root):
        dp = Path(dirpath)
        for f in sorted(files):
            if f.startswith("."):
                continue
            yield dp / f

def clean_tags(tags: List[str]) -> List[str]:
    # drop blanks + de-dupe (preserve order)
    seen = set()
    out = []
    for t in tags:
        t = (t or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

# -------------------- Image Prep --------------------
def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # Normalize orientation & mode
    image = ImageOps.exif_transpose(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Pad to square (white background)
    w, h = image.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    canvas = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    canvas.paste(image, (pad_left, pad_top))

    # Resize
    if max_dim != target_size:
        canvas = canvas.resize((target_size, target_size), Image.BICUBIC)

    # To tensor + CLIP norm
    t = TVF.pil_to_tensor(canvas).float().div(255.0)
    t = TVF.normalize(
        t,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    return t

# -------------------- Tagger --------------------
class SKTaggerVM:
    """
    STAG reimplemented to use your VisionModel for predictions.
    Handles filesystem walk, RAW/HEIC loading, and XMP writing.
    """

    def __init__(
        self,
        model_dir: Path,
        threshold: float,
        prefix: str,
        force: bool,
        test_mode: bool,
        prefer_exact_filenames: bool,
        batch_size: int = 1,
        amp_dtype: torch.dtype = torch.float16,
    ):
        register_heif_opener()

        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.tag_prefix = prefix
        self.force_tagging = force
        self.test_mode = test_mode
        self.prefer_exact_filenames = prefer_exact_filenames
        self.batch_size = max(1, int(batch_size))
        self.amp_dtype = amp_dtype

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[STAG-VM] device: {self.device}")

        # Load model
        self.model = VisionModel.load_model(str(self.model_dir))
        self.model.eval().to(self.device)

        # Determine image size from model or default
        self.image_size = getattr(self.model, "image_size", 384)

        # Load top tags
        top_path = self.model_dir / "top_tags.txt"
        with open(top_path, "r", encoding="utf-8") as f:
            self.top_tags = [ln.strip() for ln in f if ln.strip()]

        self._stop_event = threading.Event()

    # --------------- I/O ---------------
    @staticmethod
    def _should_skip(path: Path) -> bool:
        return path.suffix.lower() == ".xmp"

    @staticmethod
    def _is_raw(path: Path) -> bool:
        return path.suffix.lower() in RAW_EXTS

    def load_image(self, path: Path) -> Tuple[Optional[Image.Image], str]:
        ext = path.suffix.lower()
        if self._should_skip(path):
            return None, "none"

        img = None
        loader = "none"

        # Prefer Pillow when not RAW (HEIC/AVIF included)
        if not self._is_raw(path) or ext in NON_RAW_EXTRA:
            try:
                img = Image.open(path)
                img.load()  # force read
                loader = "pillow"
            except (UnidentifiedImageError, OSError) as e:
                print(f"[STAG-VM] Pillow failed {path.name}: {e}")

        if img is None:
            try:
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess()
                img = Image.fromarray(rgb)
                loader = "rawpy"
            except Exception as e:
                print(f"[STAG-VM] rawpy failed {path.name}: {e}")

        if img is not None and img.mode != "RGB":
            img = img.convert("RGB")

        return img, loader

    # --------------- XMP ---------------
    def is_already_tagged(self, sidecars: List[str]) -> bool:
        if self.force_tagging:
            return False
        for sc in sidecars:
            handler = XMPHandler(sc)
            if self.tag_prefix:
                if handler.has_subject_prefix(self.tag_prefix):
                    return True
            else:
                if handler.get_all_subjects():
                    return True
        return False

    def save_tags(self, image_file: Path, sidecars: List[str], tags: List[str]) -> None:
        tags = clean_tags(tags)
        if not tags:
            return

        # Create sidecar if missing
        if not sidecars:
            if self.test_mode:
                print("[STAG-VM] --test: skipping sidecar create")
                return
            sidecars = [XMPHandler.create_xmp_sidecar(str(image_file), self.prefer_exact_filenames)]

        for sc in sidecars:
            handler = XMPHandler(sc)
            before = set(handler.get_all_subjects() or [])
            for t in tags:
                if self.tag_prefix:
                    handler.add_hierarchical_subject(f"{self.tag_prefix}|{t}")
                else:
                    handler.add_hierarchical_subject(t)
            after = set(handler.get_all_subjects() or [])
            if before == after:
                print(f"[STAG-VM] no new tags for {image_file.name}")
                continue
            if not self.test_mode:
                handler.save()

    # --------------- Inference ---------------
    @torch.inference_mode()
    def _predict_probs(self, batch_t: torch.Tensor) -> torch.Tensor:
        # batch_t: [B,3,H,W] on device
        use_amp = (self.device.type == "cuda")
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_amp):
            preds = self.model({"image": batch_t})  # expects dict with 'image'
            logits = preds["tags"]                 # [B, C]
            probs = logits.sigmoid()
        return probs  # on same device

    def tags_from_probs(self, probs_1d: torch.Tensor) -> List[str]:
        # Convert 1D probs to tag list over threshold, guard for mismatched lengths
        C = min(len(self.top_tags), probs_1d.shape[-1])
        out = []
        for i in range(C):
            if float(probs_1d[i]) > self.threshold:
                out.append(self.top_tags[i])
        return out

    def predict_single(self, image: Image.Image) -> List[str]:
        t = prepare_image(image, self.image_size).unsqueeze(0).to(self.device, non_blocking=True)
        probs = self._predict_probs(t).float().cpu()[0]
        return self.tags_from_probs(probs)

    def predict_batch(self, images: List[Image.Image]) -> List[List[str]]:
        if not images:
            return []
        batch = torch.stack([prepare_image(im, self.image_size) for im in images], dim=0)
        batch = batch.to(self.device, non_blocking=True)
        probs = self._predict_probs(batch).float().cpu()  # [B,C]
        out: List[List[str]] = []
        for i in range(probs.shape[0]):
            out.append(self.tags_from_probs(probs[i]))
        return out

    # --------------- Main Walk ---------------
    def stop(self):
        self._stop_event.set()

    def enter_dir(self, root_dir: Path) -> None:
        print(f"[STAG-VM] Entering {root_dir}")
        paths = [p for p in iter_files(root_dir) if not self._should_skip(p)]
        if not paths:
            print("[STAG-VM] No files found.")
            return

        if self.batch_size <= 1:
            # simple path
            for p in paths:
                if self._stop_event.is_set():
                    print("[STAG-VM] cancelled.")
                    return

                sidecars = XMPHandler.get_xmp_sidecars_for_image(str(p))
                if self.is_already_tagged(sidecars):
                    print(f"[STAG-VM] already tagged: {p.name}")
                    continue

                img, loader = self.load_image(p)
                if img is None:
                    print(f"[STAG-VM] skip unreadable: {p.name}")
                    continue

                print(f"[STAG-VM] tagging {p} (via {loader})")
                try:
                    tags = self.predict_single(img)
                except Exception as e:
                    print(f"[STAG-VM] inference failed {p.name}: {e}")
                    continue
                finally:
                    try:
                        img.close()
                    except Exception:
                        pass

                tags = clean_tags(tags)
                print(f"[STAG-VM] tags: {tags if tags else '∅'}")
                if tags:
                    self.save_tags(p, sidecars, tags)
        else:
            # batched path
            batch_imgs: List[Image.Image] = []
            batch_paths: List[Path] = []

            def _flush():
                nonlocal batch_imgs, batch_paths
                try:
                    results = self.predict_batch(batch_imgs)
                except Exception as e:
                    print(f"[STAG-VM] batch inference failed: {e}")
                    # close all and bail this batch
                    for im in batch_imgs:
                        try: im.close()
                        except: pass
                    batch_imgs, batch_paths = [], []
                    return

                for p, im, tags in zip(batch_paths, batch_imgs, results):
                    print(f"[STAG-VM] tagging {p} (batched)")
                    tags = clean_tags(tags)
                    print(f"[STAG-VM] tags: {tags if tags else '∅'}")
                    sidecars = XMPHandler.get_xmp_sidecars_for_image(str(p))
                    if not self.is_already_tagged(sidecars) and tags:
                        self.save_tags(p, sidecars, tags)
                    try:
                        im.close()
                    except Exception:
                        pass
                batch_imgs, batch_paths = [], []

            for p in paths:
                if self._stop_event.is_set():
                    print("[STAG-VM] cancelled.")
                    return

                sidecars = XMPHandler.get_xmp_sidecars_for_image(str(p))
                if self.is_already_tagged(sidecars):
                    print(f"[STAG-VM] already tagged: {p.name}")
                    continue

                img, loader = self.load_image(p)
                if img is None:
                    print(f"[STAG-VM] skip unreadable: {p.name}")
                    continue

                batch_imgs.append(img)
                batch_paths.append(p)

                if len(batch_imgs) >= self.batch_size:
                    _flush()

            # flush tail
            if batch_imgs:
                _flush()

# -------------------- CLI --------------------
def _install_sigint_handler(tagger: SKTaggerVM):
    def _handler(signum, frame):
        print("\n[STAG-VM] SIGINT received, stopping…")
        tagger.stop()
    signal.signal(signal.SIGINT, _handler)

def parse_args():
    p = argparse.ArgumentParser(description="STAG (VisionModel) image tagger")
    p.add_argument("imagedir", metavar="DIR", help="path to images (walks recursively)")
    p.add_argument("--model-dir", default="./model_store", help="directory containing VisionModel files + top_tags.txt")
    p.add_argument("--threshold", type=float, default=0.40, help="tag probability threshold")
    p.add_argument("--prefix", default="AI", help='hierarchical subject prefix ("" for none)')
    p.add_argument("--force", action="store_true", help="retag even if already tagged")
    p.add_argument("--test", action="store_true", help="dry-run: don’t write XMP")
    p.add_argument("--prefer-exact-filenames", action="store_true",
                   help="write <name>.<ext>.xmp instead of <name>.xmp")
    p.add_argument("--batch", type=int, default=1, help="batch size (>1 for GPU throughput)")
    p.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"], help="AMP dtype on CUDA")
    return p.parse_args()

def main():
    args = parse_args()
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    tagger = SKTaggerVM(
        model_dir=Path(args.model_dir),
        threshold=args.threshold,
        prefix=args.prefix,
        force=args.force,
        test_mode=args.test,
        prefer_exact_filenames=args.prefer_exact_filenames,
        batch_size=args.batch,
        amp_dtype=amp_dtype,
    )

    _install_sigint_handler(tagger)
    tagger.enter_dir(Path(args.imagedir).expanduser().resolve())

if __name__ == "__main__":
    main()