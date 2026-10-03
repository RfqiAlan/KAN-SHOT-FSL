"""
Split NCT-CRC-HE-100K 80/20 menggunakan hard link (stratified image-level split).
Hard link: tidak menggandakan data (inode sama, storage tidak bertambah).
NCT metadata slide tidak tersedia -> image-level split (nyatakan sebagai limitasi di paper).

Verifikasi hard link: inode nct_train/ADI/img.tif == NCT-CRC-HE-100K/ADI/img.tif
"""
import os
import json
import random
from pathlib import Path

def main():
    random.seed(2021)

    src = Path("data/raw/NCT-CRC-HE-100K")
    dst_tr = Path("data/raw/nct_train")
    dst_val = Path("data/raw/nct_val")

    if not src.exists():
        print(f"Directory {src} belum ada. Pastikan dataset NCT sudah diekstrak ke sana.")
        return

    # Verifikasi ekstensi file dulu
    all_ext = set()
    for f in src.glob("**/*"):
        if f.is_file():
            all_ext.add(f.suffix.lower())
    print("Ekstensi ditemukan:", all_ext)

    VALID_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXT
        )
        assert len(images) > 0, f"Tidak ada gambar di {cls_dir}"

        random.shuffle(images)
        n_val = max(1, int(0.2 * len(images)))

        for split, imgs in [("val", images[:n_val]), ("train", images[n_val:])]:
            dst = (dst_val if split == "val" else dst_tr) / cls_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                link = dst / img.name
                if not link.exists():
                    try:
                        os.link(img, link)  # hard link - tidak copy data
                    except OSError as e:
                        if e.errno == 18:  # Cross-device link
                            raise RuntimeError(
                                "Source dan destination berbeda filesystem; "
                                "pindahkan folder ke filesystem yang sama agar hard link bekerja."
                            )
                        raise

        print(f"✅ {cls_dir.name}: {len(images)-n_val} train, {n_val} val")

    def relative_files(root: Path):
        """Menyimpan class/filename (e.g. ADI/img001.tif) untuk audit yang tidak ambigu."""
        return sorted(
            str(p.relative_to(root))
            for p in root.glob("*/*")
            if p.is_file()
        )

    # Simpan metadata split manifest untuk reproducibility
    Path("metadata").mkdir(exist_ok=True)
    split_meta = {
        "seed": 2021,
        "source": str(src),
        "split_type": "stratified image-level internal validation split",
        "train_images": relative_files(dst_tr),
        "val_images": relative_files(dst_val),
    }
    assert set(split_meta["train_images"]).isdisjoint(set(split_meta["val_images"]))
    n_total = len(split_meta["train_images"]) + len(split_meta["val_images"])
    total_src = sum(1 for p in src.glob("*/*") if p.is_file() and p.suffix.lower() in VALID_EXT)
    assert n_total == total_src, f"Total split mismatch: {n_total} vs {total_src}"
    with open("metadata/nct_split_seed2021.json", "w") as f:
        json.dump(split_meta, f, indent=2)
    print("✅ Split manifest tersimpan di metadata/nct_split_seed2021.json")

if __name__ == "__main__":
    main()
