import numpy as np
from PIL import Image
from hashdv.hashing import hash_score


def test_rotation_hash_accepts_an_exact_right_angle_copy():
    image = np.random.default_rng(19).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    rotated = Image.fromarray(image).rotate(90)
    assert hash_score(rotated, image) == 1


def test_dense_matching_distinguishes_geometric_consistency():
    import pytest
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from hashdv.geometry import verify_dense
    identity = torch.eye(64)
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(123))
    score = verify_dense(identity, torch.stack([identity, identity[permutation]]))
    assert score[0] == 1
    assert score[1] < .5
