"""
Generate episode manifest untuk 3 seed (2021, 2022, 2023) dengan dataset_root per protocol.
Semua model membaca manifest yang sama -> perbandingan fair mutlak.
Setiap protocol memiliki root dataset masing-masing (CRC-VAL vs LC25000).
Label diremap ke [0..n_way-1] lokal per episode.
"""
import json
import random
from pathlib import Path

SEEDS = [2021, 2022, 2023]

PROTOCOLS = {
    "crc_val": {
        "root": "data/raw/CRC-VAL-HE-7K",
        "n_way": 9,
        "classes": [
            "ADI", "BACK", "DEB", "LYM", "MUC",
            "MUS", "NORM", "STR", "TUM"
        ]
    },
    "lc5way": {
        "root": "data/lc25000_flat",
        "n_way": 5,
        "classes": [
            "colon_aca", "colon_n",
            "lung_aca", "lung_n", "lung_scc"
        ]
    },
    "lc_colon": {
        "root": "data/lc25000_flat",
        "n_way": 2,
        "classes": ["colon_aca", "colon_n"]
    },
    "lc_lung": {
        "root": "data/lc25000_flat",
        "n_way": 3,
        "classes": ["lung_aca", "lung_n", "lung_scc"]
    }
}

N_QUERY = 15

def main():
    Path("episodes").mkdir(exist_ok=True)

    for seed in SEEDS:
        for protocol_name, cfg in PROTOCOLS.items():
            root = Path(cfg["root"])
            if not root.exists():
                print(f"⚠️  Root {root} belum ada untuk {protocol_name}, skipping manifest generation.")
                continue

            label_map = {cls: idx for idx, cls in enumerate(cfg["classes"])}

            # Pre-load image lists to avoid reading directory 1000 times
            class_images = {}
            for cls in cfg["classes"]:
                class_dir = root / cls
                if not class_dir.exists():
                    print(f"Folder tidak ditemukan: {class_dir}")
                assert class_dir.exists(), f"Folder tidak ditemukan: {class_dir}"
                class_images[cls] = sorted([p.name for p in class_dir.iterdir() if p.is_file()])

            for shot in [1, 5, 10]:
                rng = random.Random(seed * 1000 + shot)
                episodes = []

                for ep_id in range(1000):
                    ep = {
                        "episode_id": ep_id,
                        "seed": seed,
                        "protocol": protocol_name,
                        "dataset_root": cfg["root"],
                        "shot": shot,
                        "n_way": cfg["n_way"],
                        "label_map": label_map,
                        "support": {},
                        "query": []
                    }

                    all_query = []
                    for cls in cfg["classes"]:
                        imgs = list(class_images[cls])
                        rng.shuffle(imgs)

                        support_imgs = imgs[:shot]
                        query_imgs = imgs[shot:shot + N_QUERY]

                        ep["support"][cls] = [f"{cls}/{i}" for i in support_imgs]
                        all_query += [{"path": f"{cls}/{i}", "label": label_map[cls]} for i in query_imgs]

                    ep["query"] = all_query

                    # Validasi: support dan query tidak overlap
                    support_flat = {p for imgs in ep["support"].values() for p in imgs}
                    query_paths = {q["path"] for q in ep["query"]}
                    assert support_flat.isdisjoint(query_paths), f"Overlap ep {ep_id} seed {seed}!"
                    assert all(len(v) == shot for v in ep["support"].values())

                    episodes.append(ep)

                out = f"episodes/{protocol_name}_shot{shot}_seed{seed}.json"
                with open(out, "w") as f:
                    json.dump(episodes, f, indent=2)
                print(f"✅ {protocol_name} shot={shot} seed={seed}: 1000 episodes -> {out}")

if __name__ == "__main__":
    main()
