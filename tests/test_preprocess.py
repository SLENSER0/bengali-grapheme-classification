import numpy as np

from bengali.data import crop_resize
from bengali.utils import RAW_H, RAW_W


def test_crop_resize_shape_and_centering():
    img = np.full((RAW_H, RAW_W), 250, dtype=np.uint8)  # white paper
    img[40:90, 150:200] = 10  # dark ink blob, off-centre to the right
    out, st = crop_resize(img, size=128)
    assert out.shape == (128, 128) and out.dtype == np.uint8
    # ink becomes bright and ends up near the centre after cropping
    ys, xs = np.nonzero(out > 128)
    assert abs(ys.mean() - 64) < 6 and abs(xs.mean() - 64) < 6
    assert out[0, 0] == 0  # background is zero
    assert st["bbox_h"] >= 50 and st["bbox_w"] >= 50


def test_border_artifacts_are_ignored():
    img = np.full((RAW_H, RAW_W), 250, dtype=np.uint8)
    img[:2, :] = 0  # scanner line along the top border
    img[60:80, 100:120] = 0
    _, st = crop_resize(img)
    assert st["bbox_h"] < 60  # the box is around the blob, not stretched to the border line


def test_empty_image_does_not_crash():
    out, _ = crop_resize(np.full((RAW_H, RAW_W), 255, dtype=np.uint8), size=64)
    assert out.shape == (64, 64)


def test_read_parquet_ignores_pandas_index_column(tmp_path):
    """The competition files carry an extra '__index_level_0__' column next to the 32332 pixel columns."""
    import pandas as pd

    from bengali.data import read_parquet_images

    rng = np.random.default_rng(0)
    pix = rng.integers(0, 256, size=(3, RAW_H * RAW_W), dtype=np.uint8)
    df = pd.DataFrame(pix, columns=[str(i) for i in range(RAW_H * RAW_W)])
    df.insert(0, "image_id", [f"Train_{i}" for i in range(3)])
    df.index = [10, 11, 12]  # non-default index -> pyarrow stores it as __index_level_0__
    df.to_parquet(tmp_path / "x.parquet")
    ids, imgs = read_parquet_images(tmp_path / "x.parquet")
    assert list(ids) == ["Train_0", "Train_1", "Train_2"]
    assert imgs.shape == (3, RAW_H, RAW_W)
    assert (imgs.reshape(3, -1) == pix).all()


def test_near_duplicates_finds_copies_within_grapheme_only():
    from bengali.eda import near_duplicates

    rng = np.random.default_rng(1)
    imgs = rng.integers(0, 256, size=(6, 128, 128), dtype=np.uint8)
    imgs[3] = np.clip(imgs[0].astype(int) + rng.integers(-3, 4, size=(128, 128)), 0, 255)  # near copy of 0
    imgs[4] = imgs[1]  # exact copy of 1 but labelled as another grapheme -> must not be paired
    graphemes = np.array(["a", "b", "a", "a", "c", "c"])
    nd = near_duplicates(imgs, graphemes)
    assert set(map(tuple, nd[["row_a", "row_b"]].to_numpy())) == {(0, 3)}
