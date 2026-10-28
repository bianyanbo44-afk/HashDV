"""The paper's 64-bit, six-view rotation-aware perceptual hash."""
import imagehash
from PIL import Image


def variants(image):
    image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    # PIL.rotate(expand=False), rather than transpose, preserves the original rule for non-square images.
    views = (image, image.rotate(90), image.rotate(180), image.rotate(270),
             image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), image.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    return [imagehash.phash(view) for view in views]


def reference_hash(image):
    return imagehash.phash(image if isinstance(image, Image.Image) else Image.fromarray(image))


def hash_score(image, reference):
    return 1.0 - min(code - reference_hash(reference) for code in variants(image)) / 64.0
