import cv2
import numpy as np
import math
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from segearthov3_segmentor import SegEarthOV3Segmentation


PLAN_ROOT = Path(__file__).resolve().parent
SEARCH_ROOT = PLAN_ROOT / "data" / "MAR20_seg" / "train_sub" / "images"
LABEL_ROOT = PLAN_ROOT / "data" / "MAR20_seg" / "train_sub" / "labelTxt"
OUTPUT_DIR = PLAN_ROOT / "outputs" / "generated_labels"
CHECKPOINT_PATH = PLAN_ROOT / "models" / "sam3.pt"
BPE_PATH = PLAN_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"


IMAGES_PER_CLASS = 150

MAX_ITEMS_PER_IMAGE = 5


USE_STRUCT = True
USE_SCALE = True


STRUCT_SIGMAS = (2.0, 5.0, 10.0)
SCALE_N = 4
SCALE_RANGE = (0.5, 1.25)
SCALE_WEIGHTS = (0.1, 0.2, 0.4, 0.3)
COVERAGE_THD = 0.95

length_ratios_rel_to_f16 = [
    1.45,
    1.98,
    3.52,
    5.00,
    1.00,
    3.59,
    3.09,
    3.22,
    2.36,
    2.95,
    3.09,
    2.82,
    1.29,
    2.76,
    1.25,
    1.22,
    3.29,
    3.67,
    1.55,
    1.63,
]
aspect_ratios = [
    1.43,
    0.74,
    1.02,
    1.11,
    1.51,
    0.97,
    1.05,
    0.86,
    1.17,
    1.06,
    1.05,
    1.24,
    1.49,
    1.04,
    1.39,
    1.34,
    0.97,
    1.10,
    1.59,
    1.39,
]
plane_names = [f"A{i}" for i in range(1, 21)]
plane_name_to_idx = {name: i for i, name in enumerate(plane_names)}

NAME_LIST_PATH = PLAN_ROOT / "configs" / "mar20_names.txt"
TARGET_CLS_INDICES = [1]


def setup_model():
    if not NAME_LIST_PATH.is_file():
        raise FileNotFoundError(f"Class-name config not found: {NAME_LIST_PATH}")
    model = SegEarthOV3Segmentation(
        classname_path=NAME_LIST_PATH,
        prob_thd=0.1,
        confidence_threshold=0.1,
        slide_stride=512,
        slide_crop=512,
        checkpoint_path=CHECKPOINT_PATH,
        bpe_path=BPE_PATH,
    )
    return model


def calc_scale_for_image(label_path):
    if not label_path.exists():
        return None
    base_lengths = []
    all_lengths = []

    with label_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            poly = list(map(float, parts[:8]))
            cls_name = parts[8]
            d1 = math.hypot(poly[0] - poly[2], poly[1] - poly[3])
            d2 = math.hypot(poly[2] - poly[4], poly[3] - poly[5])
            length = max(d1, d2)

            if cls_name in plane_name_to_idx:
                idx = plane_name_to_idx[cls_name]
                ratio = length_ratios_rel_to_f16[idx]
                base_lengths.append(length / ratio)

            all_lengths.append(length)
        except Exception:
            continue

    if base_lengths:
        return np.mean(base_lengths)
    elif all_lengths:
        return np.median(all_lengths) / np.mean(length_ratios_rel_to_f16)
    else:
        return None


def get_plane_dims_dynamic(idx, base_f16_px):
    length = length_ratios_rel_to_f16[idx] * base_f16_px
    wingspan = length / aspect_ratios[idx]
    return int(wingspan), int(length)


def compute_A_geo(M_valid):

    D = cv2.distanceTransform(M_valid, cv2.DIST_L2, 5)
    A_geo = D / (D.max() + 1e-8)
    return A_geo, D


def compute_structure_field(I_gray, sigmas=STRUCT_SIGMAS):

    I = I_gray.astype(np.float32)
    Ix = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3)
    Jxx_r, Jxy_r, Jyy_r = Ix * Ix, Ix * Iy, Iy * Iy

    Sxx = np.zeros_like(I)
    Sxy = np.zeros_like(I)
    Syy = np.zeros_like(I)
    for s in sigmas:
        Sxx += cv2.GaussianBlur(Jxx_r, (0, 0), s)
        Sxy += cv2.GaussianBlur(Jxy_r, (0, 0), s)
        Syy += cv2.GaussianBlur(Jyy_r, (0, 0), s)

    tmp = np.sqrt((Sxx - Syy) ** 2 + 4.0 * Sxy**2)
    lam1 = 0.5 * (Sxx + Syy + tmp)
    lam2 = 0.5 * (Sxx + Syy - tmp)

    theta_grad = 0.5 * np.arctan2(2.0 * Sxy, Sxx - Syy)
    theta_tex = theta_grad + np.pi / 2.0

    theta_tex = (theta_tex + np.pi / 2.0) % np.pi - np.pi / 2.0

    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-8)
    return theta_tex.astype(np.float32), coherence.astype(np.float32)


def compute_A_scale(
    M_valid, W_c, H_c, n_scales=SCALE_N, scale_range=SCALE_RANGE, weights=SCALE_WEIGHTS
):

    base_r = int(np.ceil(min(W_c, H_c) / 2.0))
    scales = np.linspace(scale_range[0], scale_range[1], n_scales)
    w_arr = np.asarray(weights, dtype=np.float32)
    A = np.zeros(M_valid.shape, dtype=np.float32)
    for s, w in zip(scales, w_arr):
        r = max(1, int(np.ceil(base_r * s)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        M_e = cv2.erode(M_valid, kernel).astype(np.float32)
        A += w * M_e
    return A / (w_arr.sum() + 1e-8)


def compute_affordance_field(A_geo, A_struct, A_scale):

    return A_geo * A_struct * A_scale


def find_local_maxima(field, box_w, box_h):

    kernel_size = max(3, min(box_w, box_h) // 2)
    dilated = cv2.dilate(field, np.ones((kernel_size, kernel_size), np.float32))
    local_max = (field == dilated) & (field > 0)
    ys, xs = np.where(local_max)
    if len(xs) == 0:
        return []
    scores = field[ys, xs]
    order = np.argsort(scores)[::-1]
    return list(zip(ys[order], xs[order]))


def _best_fitting_angle(
    raw_mask,
    occupied_mask,
    dist_map,
    cx,
    cy,
    box_w,
    box_h,
    cand_angles,
    min_cov=COVERAGE_THD,
):

    best_ang = None
    best_score = -float("inf")
    for ang in cand_angles:
        rect = ((float(cx), float(cy)), (float(box_w), float(box_h)), float(ang))
        pts = np.int32(cv2.boxPoints(rect))
        temp = np.zeros_like(raw_mask)
        cv2.fillPoly(temp, [pts], 1)
        if np.any(cv2.bitwise_and(occupied_mask, temp)):
            continue
        mask_pixels = int(np.sum(temp))
        if mask_pixels == 0:
            continue
        covered = int(np.sum(cv2.bitwise_and(raw_mask, temp)))
        if covered >= mask_pixels * min_cov:
            score = -np.std(dist_map[temp == 1])
            if score > best_score:
                best_score = score
                best_ang = ang
    return best_ang


def _legacy_orientation_angles(dist_map, cx, cy, h_orig, w_orig):

    win = 5
    x1, y1 = max(0, cx - win), max(0, cy - win)
    x2, y2 = min(w_orig, cx + win), min(h_orig, cy + win)
    local_patch = dist_map[y1:y2, x1:x2]
    if local_patch.size == 0:
        return None
    sobelx = cv2.Sobel(local_patch, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(local_patch, cv2.CV_64F, 0, 1, ksize=5)
    grad_angle = np.degrees(np.arctan2(np.mean(sobely), np.mean(sobelx)))
    return [grad_angle, grad_angle + 90, grad_angle + 180, grad_angle + 270]


def find_placements_aap(
    raw_mask, occupied_mask, theta_tex, coherence, box_w, box_h, max_count
):

    results = []
    h_orig, w_orig = raw_mask.shape

    A_geo, dist_map = compute_A_geo(raw_mask)
    if USE_SCALE:
        A_scale = compute_A_scale(raw_mask, box_w, box_h)
    else:
        A_scale = np.ones_like(A_geo, dtype=np.float32)
    A_pos = A_geo * A_scale

    anchors = find_local_maxima(A_pos, box_w, box_h)
    if not anchors:
        return []

    if USE_STRUCT and coherence is not None:
        a_total = [A_pos[y, x] * coherence[y, x] for (y, x) in anchors]
        anchors = [anchors[i] for i in np.argsort(a_total)[::-1]]

    candidate = anchors[: max_count * 20]

    for cy, cx in candidate:
        if len(results) >= max_count:
            break
        cy, cx = int(cy), int(cx)
        if occupied_mask[cy, cx] > 0:
            continue

        if USE_STRUCT and theta_tex is not None:

            phi = math.degrees(float(theta_tex[cy, cx]))
            cand_angles = [phi + 90.0]
        else:
            cand_angles = _legacy_orientation_angles(dist_map, cx, cy, h_orig, w_orig)
            if cand_angles is None:
                continue

        best_ang = _best_fitting_angle(
            raw_mask, occupied_mask, dist_map, cx, cy, box_w, box_h, cand_angles
        )
        if best_ang is None:
            continue

        results.append((float(cx), float(cy), float(best_ang)))

        rect = ((float(cx), float(cy)), (float(box_w), float(box_h)), float(best_ang))
        pts = np.int32(cv2.boxPoints(rect))
        cv2.fillPoly(occupied_mask, [pts], 1)
        cv2.fillPoly(raw_mask, [pts], 0)

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_files = sorted(SEARCH_ROOT.glob("*.png")) + sorted(SEARCH_ROOT.glob("*.jpg"))

    if not img_files:
        print(f"No images found in {SEARCH_ROOT}")
        return

    print(f"Found {len(img_files)} images in {SEARCH_ROOT}")
    print(f"[AAP] USE_STRUCT={USE_STRUCT}, USE_SCALE={USE_SCALE}")
    model = setup_model()

    print("Phase 1: Pre-processing images (Segmentation & Structure field)...")
    image_cache = {}
    valid_img_files = []

    for img_path in tqdm(img_files):
        label_path = LABEL_ROOT / f"{img_path.stem}.txt"

        current_base_px = calc_scale_for_image(label_path)
        if current_base_px is None:
            continue

        img = Image.open(img_path).convert("RGB")
        seg_mask = model.predict(img)

        raw_mask = np.zeros_like(seg_mask, dtype=np.uint8)
        for idx in TARGET_CLS_INDICES:
            raw_mask = cv2.bitwise_or(raw_mask, (seg_mask == idx).astype(np.uint8))

        if np.sum(raw_mask) < 2000:
            continue

        global_occupied_mask = np.zeros_like(raw_mask)
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        poly = (
                            np.array(list(map(float, parts[:8])))
                            .reshape(-1, 2)
                            .astype(np.int32)
                        )
                        cv2.fillPoly(global_occupied_mask, [poly], 1)
                        cv2.fillPoly(raw_mask, [poly], 0)

        if USE_STRUCT:
            gray = np.array(img.convert("L"))
            theta_tex, coherence = compute_structure_field(gray)
        else:
            theta_tex, coherence = None, None

        image_cache[img_path] = {
            "scale": current_base_px,
            "raw_mask": raw_mask,
            "occupied_mask": global_occupied_mask,
            "theta_tex": theta_tex,
            "coherence": coherence,
        }
        valid_img_files.append(img_path)

    print(f"Pre-processing done. Valid images: {len(valid_img_files)}")

    print("Phase 2: Generating placements by class (AAP)...")

    for idx, name in enumerate(plane_names):
        print(f"Processing Class {name}...")

        class_out_path = OUTPUT_DIR / f"{name}.txt"
        f_cls = class_out_path.open("w", encoding="utf-8")
        f_cls.write("image_path,class_name,cx,cy,w,h,angle\n")

        random.shuffle(valid_img_files)
        images_processed_count = 0

        for img_path in tqdm(valid_img_files, desc=f"Class {name}"):
            if images_processed_count >= IMAGES_PER_CLASS:
                break

            cache = image_cache[img_path]
            current_base_px = cache["scale"]

            raw_mask = cache["raw_mask"].copy()
            occupied_mask = cache["occupied_mask"].copy()

            w, h = get_plane_dims_dynamic(idx, current_base_px)

            placements = find_placements_aap(
                raw_mask,
                occupied_mask,
                cache["theta_tex"],
                cache["coherence"],
                w,
                h,
                max_count=MAX_ITEMS_PER_IMAGE,
            )

            if len(placements) > 0:
                for cx, cy, angle in placements:
                    relative_img_path = Path(img_path).relative_to(PLAN_ROOT).as_posix()
                    f_cls.write(
                        f"{relative_img_path},{name},{cx:.2f},{cy:.2f},{w},{h},{angle:.2f}\n"
                    )
                f_cls.flush()
                images_processed_count += 1

        f_cls.close()
        print(
            f"Class {name}: Inserted into {images_processed_count} images. Saved to {class_out_path}"
        )

    print("All done!")


if __name__ == "__main__":
    main()
