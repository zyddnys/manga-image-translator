import json
import os
from pathlib import Path
import cv2
import pytest
import numpy as np
from manga_translator.detection import dispatch, unload
from utils import load_rgb_image


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testdata/detection/output")

def make_run_id(detector, image_path, detect_size) -> str:
    # 例: "default__op-1178-2-raw__2048"
    stem = Path(image_path).stem
    return f"{detector}__{stem}__{detect_size}"

async def run_detection(image, params):
    detector_key = params['detector_key']
    try:
        textlines, raw_mask, mask = await dispatch(
            detector_key,
            image,
            params['detect_size'],
            params['text_threshold'],
            params['box_threshold'],
            params['unclip_ratio'],
            params['invert'],
            params['gamma_correct'],
            params['rotate'],
            params['auto_rotate'],
            device=params['device'],
            verbose=params['verbose'],
        )
        return textlines, raw_mask, mask
    finally:
        await unload(detector_key)



def _save_rgb_png(path: Path, image_rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

def _align_mask_to_image(mask: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

def save_review_artifacts(run_id: str, image_rgb: np.ndarray, textlines: list, raw_mask: np.ndarray, mask: np.ndarray | None):
    run_dir = Path(OUTPUT_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_rgb_png(run_dir / "input.png", image_rgb)

    bboxes = np.copy(image_rgb)
    for txtln in textlines:
        cv2.polylines(bboxes, [txtln.pts], True, color=(255, 0, 0), thickness=2)
    _save_rgb_png(run_dir / "bboxes.png", bboxes)

    raw_mask_aligned = _align_mask_to_image(raw_mask, image_rgb)
    cv2.imwrite(str(run_dir / "mask_raw.png"), raw_mask_aligned)
    if mask is not None:
        cv2.imwrite(str(run_dir / "mask.png"), _align_mask_to_image(mask, image_rgb))

    overlay = image_rgb.copy()
    active = raw_mask_aligned > 0
    overlay[active] = (overlay[active] * 0.6 + np.array([255, 0, 0], dtype=np.uint8) * 0.4).astype(np.uint8)
    _save_rgb_png(run_dir / "overlay.png", overlay)

    summary = {
        "textline_count": len(textlines),
        "image_shape": list(image_rgb.shape),
        "raw_mask_shape": list(raw_mask.shape),
        "raw_mask_nonzero_ratio": float((raw_mask_aligned > 0).mean()),
        "textlines": [
            {"pts": q.pts.tolist(), "prob": float(q.prob), "area": int(q.area)}
            for q in textlines
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return run_dir

def print_review_summary(run_dir, textlines, raw_mask):
    print(f"Output: {run_dir}")
    print(f"Textlines: {len(textlines)}")
    print(f"Mask nonzero: {(raw_mask > 0).sum()} px")
    for i, q in enumerate(textlines[:10]):
        print(f"  [{i}] prob={q.prob:.3f} area={q.area} pts={q.pts.tolist()}")
    if len(textlines) > 10:
        print(f"  ... and {len(textlines) - 10} more")


@pytest.mark.asyncio
async def test_detection_manual_review(detector, image_path, detection_params):
    print(f"detector: {detector},\n image: {image_path},\n params: {detection_params}")
    if detector is None:
        pytest.skip("pass, --detector default")
    
    assert Path(image_path).exists(), f"missing fixture: {image_path}"

    image = load_rgb_image(image_path)
    run_id = make_run_id(detector, image_path, detection_params["detect_size"])
    textlines, raw_mask, mask = await run_detection(image, detection_params)
    
    run_dir = save_review_artifacts(run_id, image, textlines, raw_mask, mask)
    print_review_summary(run_dir, textlines, raw_mask)

