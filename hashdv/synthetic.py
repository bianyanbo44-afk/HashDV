"""Deterministic crop/rotation and compound corruption used for the confirmation copies."""
import hashlib
import math
import numpy as np


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def corrupt(image, operators, severity, seed):
    import cv2
    rng = np.random.default_rng(seed)
    x = np.asarray(image, dtype=np.uint8).copy()
    if not 1 <= severity <= 5:
        raise ValueError("Severity must be in 1..5")
    for name in operators:
        y = x.astype(np.float32)
        if name == "gaussian_noise":
            y = y + rng.normal(0, (4, 8, 14, 22, 32)[severity - 1], size=(*y.shape[:2], 1))
        elif name == "gaussian_blur":
            sigma = (.45, .75, 1.15, 1.65, 2.30)[severity - 1]
            y = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma, sigmaY=sigma)
        elif name == "brightness":
            y *= (.82, .68, .54, .40, .28)[severity - 1]
        elif name == "contrast":
            mean = y.mean(axis=(0, 1), keepdims=True)
            y = mean + (.82, .68, .54, .40, .28)[severity - 1] * (y - mean)
        elif name == "jpeg":
            bgr = cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, (70, 50, 35, 22, 12)[severity - 1]])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            y = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
        else:
            raise ValueError(f"Unknown operator: {name}")
        x = np.clip(y, 0, 255).astype(np.uint8)
    return x


def make_copy(image, rng, operators, severity):
    import cv2
    h, w = image.shape[:2]
    if min(h, w) < 8:
        raise ValueError("Source image must be at least 8 pixels on each side")
    area = float(rng.uniform(.70, .95))
    cw, ch = max(8, int(round(w * math.sqrt(area)))), max(8, int(round(h * math.sqrt(area))))
    x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    geometry = cv2.resize(image[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    angle = float(rng.uniform(-3, 3))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    geometry = cv2.warpAffine(geometry, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    result = corrupt(geometry, operators, severity, int(rng.integers(0, 2**31 - 1)))
    return result, {"u": round(area, 4), "x0": x, "y0": y, "angle": round(angle, 3),
                    "ops": list(operators), "sev": severity}
