from pathlib import Path
import csv
import numpy as np
from PIL import Image


ROOT = Path("/home/shubhra/RMIA_IN_CXR/data/IN-CXR")
OUT = Path("/home/shubhra/RMIA_IN_CXR/exp/incxr_16")

SEED = 314159265

NUM_REFS = 16
IN_REFS_PER_SAMPLE = 8


def get_label(path):
    cls = path.parent.name.lower()

    if cls == "normal":
        return 0

    if cls == "abnormal":
        return 1

    raise ValueError(f"Unknown class folder: {path.parent.name}")


def valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception as e:
        print("INVALID:", path, e)
        return False


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        list((ROOT / "Normal").glob("*"))
        +
        list((ROOT / "AbNormal").glob("*"))
    )

    paths = [
        p for p in paths
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    print("Images found:", len(paths))

    records = []
    invalid = []

    for i, p in enumerate(paths):

        if (i + 1) % 1000 == 0:
            print(f"Checking {i+1}/{len(paths)}")

        if not valid_image(p):
            invalid.append(p)
            continue

        label = get_label(p)

        records.append(
            {
                "relative_path": p.relative_to(ROOT).as_posix(),
                "label": label,
            }
        )

    # Fixed immutable candidate order
    records = sorted(
        records,
        key=lambda x: x["relative_path"]
    )

    for i, r in enumerate(records):
        r["candidate_id"] = i

    N = len(records)

    labels = np.array(
        [r["label"] for r in records],
        dtype=np.int64
    )

    # -------------------------------------------------
    # Save candidates
    # -------------------------------------------------

    with open(
        OUT / "candidates.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "relative_path",
                "label"
            ]
        )

        writer.writeheader()

        for r in records:
            writer.writerow(r)

    np.save(
        OUT / "labels.npy",
        labels
    )

    # -------------------------------------------------
    # Reference membership
    # -------------------------------------------------

    rng = np.random.default_rng(SEED)

    keep = np.zeros(
        (NUM_REFS, N),
        dtype=bool
    )

    for i in range(N):

        chosen = rng.choice(
            NUM_REFS,
            size=IN_REFS_PER_SAMPLE,
            replace=False
        )

        keep[chosen, i] = True

    np.save(
        OUT / "keep.npy",
        keep
    )

    # -------------------------------------------------
    # Target membership
    # exactly half of each class
    # -------------------------------------------------

    target_keep = np.zeros(
        N,
        dtype=bool
    )

    for cls in [0, 1]:

        idx = np.where(
            labels == cls
        )[0]

        idx = rng.permutation(idx)

        half = len(idx) // 2

        target_keep[
            idx[:half]
        ] = True

    np.save(
        OUT / "target_keep.npy",
        target_keep
    )

    # -------------------------------------------------
    # Invalid list
    # -------------------------------------------------

    with open(
        OUT / "invalid_images.txt",
        "w"
    ) as f:

        for p in invalid:
            f.write(str(p) + "\n")

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("IN-CXR DATASET")
    print("=" * 60)

    print("Candidates :", N)
    print("Normal     :", int((labels == 0).sum()))
    print("AbNormal   :", int((labels == 1).sum()))
    print("Invalid    :", len(invalid))

    print()
    print("=" * 60)
    print("REFERENCE BANK")
    print("=" * 60)

    print("keep shape :", keep.shape)

    print(
        "IN refs min/max/mean:",
        keep.sum(0).min(),
        keep.sum(0).max(),
        keep.sum(0).mean()
    )

    print(
        "OUT refs min/max/mean:",
        (~keep).sum(0).min(),
        (~keep).sum(0).max(),
        (~keep).sum(0).mean()
    )

    print()
    print("Reference train sizes:")

    for r in range(NUM_REFS):

        n_total = int(
            keep[r].sum()
        )

        n_normal = int(
            (
                keep[r]
                &
                (labels == 0)
            ).sum()
        )

        n_abnormal = int(
            (
                keep[r]
                &
                (labels == 1)
            ).sum()
        )

        print(
            f"ref {r:02d}: "
            f"{n_total} "
            f"(N={n_normal}, A={n_abnormal})"
        )

    print()
    print("=" * 60)
    print("TARGET")
    print("=" * 60)

    print(
        "Members:",
        int(target_keep.sum())
    )

    print(
        "Nonmembers:",
        int((~target_keep).sum())
    )

    print(
        "Member Normal:",
        int(
            (
                target_keep
                &
                (labels == 0)
            ).sum()
        )
    )

    print(
        "Member AbNormal:",
        int(
            (
                target_keep
                &
                (labels == 1)
            ).sum()
        )
    )

    print(
        "Nonmember Normal:",
        int(
            (
                (~target_keep)
                &
                (labels == 0)
            ).sum()
        )
    )

    print(
        "Nonmember AbNormal:",
        int(
            (
                (~target_keep)
                &
                (labels == 1)
            ).sum()
        )
    )

    assert keep.shape == (16, N)

    assert np.all(
        keep.sum(axis=0) == 8
    )

    assert np.all(
        (~keep).sum(axis=0) == 8
    )

    print()
    print("Saved to:")
    print(OUT)


if __name__ == "__main__":
    main()