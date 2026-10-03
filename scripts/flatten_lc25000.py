"""
Flatten LC25000 ke satu level menggunakan hard link:
data/lc25000/{colon_image_sets,lung_image_sets}/* -> data/lc25000_flat/{colon_aca, colon_n, lung_aca, lung_n, lung_scc}
"""
import os
from pathlib import Path

def main():
    src = Path("data/lc25000")
    dst = Path("data/lc25000_flat")
    dst.mkdir(exist_ok=True)

    if not src.exists():
        print(f"Directory {src} belum ada.")
        return

    counts = {}
    for class_folder in sorted(src.glob("*/*")):
        if not class_folder.is_dir():
            continue
        imgs = [p for p in class_folder.iterdir() if p.is_file()]
        assert len(imgs) > 0, f"Folder kosong: {class_folder}"
        target = dst / class_folder.name
        target.mkdir(exist_ok=True)
        for img in imgs:
            link = target / img.name
            if not link.exists():
                try:
                    os.link(img, link)
                except OSError as e:
                    if e.errno == 18:
                        raise RuntimeError(
                            "Source dan destination berbeda filesystem; "
                            "pindahkan folder ke filesystem yang sama agar hard link bekerja."
                        )
                    raise
        counts[class_folder.name] = len(imgs)
        print(f"✅ {class_folder.name}: {len(imgs)} gambar")

    assert len(counts) == 5, f"Harusnya 5 kelas, didapat {len(counts)}: {list(counts.keys())}"
    print(f"\n✅ Total: {len(counts)} kelas, {sum(counts.values())} gambar berhasil di-flatten.")

if __name__ == "__main__":
    main()
