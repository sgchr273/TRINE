
import os
import json
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import Subset
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Any
import glob
from scipy.io import loadmat
import csv
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize_class_name(name: str) -> str:
    name = str(name).replace("_", " ").replace("-", " ").strip()
    return name

def infer_class_names(dataset) -> List[str]:
    """
    Generic class name inference for torchvision-style datasets.
    Tries common attributes in a robust order.
    """
    if hasattr(dataset, "classes") and dataset.classes is not None:
        return [normalize_class_name(c) for c in list(dataset.classes)]

    if hasattr(dataset, "class_to_idx") and dataset.class_to_idx is not None:
        items = sorted(dataset.class_to_idx.items(), key=lambda x: x[1])
        return [normalize_class_name(k) for k, _ in items]

    if hasattr(dataset, "categories") and dataset.categories is not None:
        return [normalize_class_name(c) for c in list(dataset.categories)]

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        unique_labels = sorted(set(int(x) for x in targets))
        return [f"class {i}" for i in unique_labels]

    raise ValueError(
        "Could not infer class names from dataset. "
        "Please add dataset-specific handling."
    )

class JSONSplitImageDataset(Dataset):
    def __init__(self, dataset_root, image_roots, split="test", transform=None):
        self.dataset_root = dataset_root
        self.image_roots = image_roots
        self.split = split
        self.transform = transform

        if isinstance(self.image_roots, str):
            self.image_roots = [self.image_roots]

        split_file = self._find_json_split_file(dataset_root)

        with open(split_file, "r") as f:
            data = json.load(f)

        if split not in data:
            raise KeyError(
                f"Split '{split}' not found in {split_file}. "
                f"Available keys: {list(data.keys())}"
            )

        # Build class list from all available splits, not only current split.
        label_to_class = {}

        for split_name, split_entries in data.items():
            if not isinstance(split_entries, list):
                continue

            for item in split_entries:
                rel_path, label, class_name = item[0], int(item[1]), str(item[2])
                label_to_class[label] = class_name

        self.classes = [label_to_class[i] for i in sorted(label_to_class.keys())]
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

        self.samples = []
        self.targets = []

        for item in data[split]:
            rel_path = str(item[0]).replace("\\", "/").lstrip("./")
            target = int(item[1])

            image_path = self._find_image_path(rel_path)
            if image_path is None:
                print(f"Skipping missing image: {rel_path}")
                continue

            self.samples.append((image_path, target))
            self.targets.append(target)

        self.imgs = self.samples

        print(
            f"Loaded {len(self.samples)} samples from split='{split}' "
            f"using split file: {split_file}"
        )

    def _find_json_split_file(self, dataset_root):
        for fname in os.listdir(dataset_root):
            if not fname.endswith(".json"):
                continue

            path = os.path.join(dataset_root, fname)

            try:
                with open(path, "r") as f:
                    data = json.load(f)

                if isinstance(data, dict) and "train" in data and "test" in data:
                    return path
            except Exception:
                pass

        raise FileNotFoundError(
            f"No JSON split file with train/test keys found in {dataset_root}"
        )

    def _find_image_path(self, rel_path):
        rel_path = str(rel_path).replace("\\", "/").lstrip("./")
        base_name = os.path.basename(rel_path)

        parts = rel_path.split("/")

        # Stanford Cars strict handling.
        # If JSON says cars_test/00001.jpg, only search cars_test.
        # If JSON says cars_train/00001.jpg, only search cars_train.
        if len(parts) >= 2 and parts[0] in ["cars_train", "cars_test"]:
            split_folder = parts[0]

            strict_candidates = [
                os.path.join(self.dataset_root, rel_path),
                os.path.join(self.dataset_root, split_folder, split_folder, *parts[1:]),
                os.path.join(self.dataset_root, split_folder, base_name),
            ]

            strict_candidates = list(dict.fromkeys(strict_candidates))

            for path in strict_candidates:
                if os.path.isfile(path):
                    return path

            raise FileNotFoundError(
                "Could not resolve split specific Stanford Cars image path.\n"
                f"Relative path from JSON: {rel_path}\n"
                f"Dataset root: {self.dataset_root}\n"
                f"Checked candidates:\n" +
                "\n".join(strict_candidates)
            )

        # Generic fallback for other JSON datasets.
        candidates = []

        stem, ext = os.path.splitext(base_name)

        possible_exts = [
            ext,
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
        ]

        possible_exts = list(dict.fromkeys([e for e in possible_exts if e]))

        for image_root in self.image_roots:
            candidates.append(os.path.join(image_root, rel_path))
            candidates.append(os.path.join(image_root, base_name))

            rel_dir = os.path.dirname(rel_path)

            for new_ext in possible_exts:
                candidates.append(os.path.join(image_root, rel_dir, stem + new_ext))
                candidates.append(os.path.join(image_root, stem + new_ext))

        candidates = list(dict.fromkeys(candidates))

        for path in candidates:
            if os.path.isfile(path):
                return path

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


class StanfordCarsZhouDataset(Dataset):
    """
    Stanford Cars loader using split_zhou_StanfordCars.json.

    Expected structure:
        stanford-cars/
            car_devkit/
            cars_train/
                cars_train/        optional nested format
                    00001.jpg
                00001.jpg          optional flat format
            cars_test/
                cars_test/         optional nested format
                    00001.jpg
                00001.jpg          optional flat format
            split_zhou_StanfordCars.json

    This loader uses the JSON file as the source of truth for:
        image path
        label
        class name

    It also handles both 0-based and 1-based labels safely.
    """

    def __init__(self, dataset_root, split="test", transform=None):
        self.dataset_root = os.path.abspath(dataset_root)
        self.split = split
        self.transform = transform

        split_file = os.path.join(self.dataset_root, "split_zhou_StanfordCars.json")

        if not os.path.isfile(split_file):
            # fallback: find any StanfordCars split json
            candidates = [
                os.path.join(self.dataset_root, f)
                for f in os.listdir(self.dataset_root)
                if f.endswith(".json") and "StanfordCars" in f
            ]
            if len(candidates) == 0:
                raise FileNotFoundError(
                    f"Could not find split_zhou_StanfordCars.json in:\n{self.dataset_root}"
                )
            split_file = candidates[0]

        with open(split_file, "r") as f:
            data = json.load(f)

        if split not in data:
            raise KeyError(
                f"Split '{split}' not found in {split_file}. "
                f"Available splits: {list(data.keys())}"
            )

        # -------------------------------------------------
        # Build class mapping from the full JSON file.
        # This is critical because the JSON label order is
        # the true order used by the split.
        # -------------------------------------------------
        raw_label_to_class = {}

        for split_name, entries in data.items():
            if not isinstance(entries, list):
                continue

            for item in entries:
                rel_path = str(item[0])
                raw_label = int(item[1])
                class_name = normalize_class_name(str(item[2]))

                if raw_label in raw_label_to_class:
                    if raw_label_to_class[raw_label] != class_name:
                        raise ValueError(
                            f"Conflicting class names for label {raw_label}: "
                            f"'{raw_label_to_class[raw_label]}' vs '{class_name}'"
                        )

                raw_label_to_class[raw_label] = class_name

        sorted_raw_labels = sorted(raw_label_to_class.keys())
        num_classes = len(sorted_raw_labels)

        # Handle both common cases:
        #   labels are 0..195
        #   labels are 1..196
        if sorted_raw_labels == list(range(num_classes)):
            raw_to_target = {raw: raw for raw in sorted_raw_labels}
        elif sorted_raw_labels == list(range(1, num_classes + 1)):
            raw_to_target = {raw: raw - 1 for raw in sorted_raw_labels}
        else:
            # General safe fallback
            raw_to_target = {raw: i for i, raw in enumerate(sorted_raw_labels)}

        self.classes = [
            raw_label_to_class[raw]
            for raw in sorted_raw_labels
        ]

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }

        # -------------------------------------------------
        # Load split samples
        # -------------------------------------------------
        self.samples = []
        self.targets = []

        for item in data[split]:
            rel_path = str(item[0]).replace("\\", "/").lstrip("./")
            raw_label = int(item[1])

            if raw_label not in raw_to_target:
                raise KeyError(f"Raw label {raw_label} was not found in class mapping.")

            target = raw_to_target[raw_label]
            image_path = self._resolve_image_path(rel_path, split)

            self.samples.append((image_path, target))
            self.targets.append(target)

        self.imgs = self.samples

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No Stanford Cars samples loaded for split='{split}'. "
                f"Check JSON file and image folders."
            )

        if min(self.targets) < 0 or max(self.targets) >= len(self.classes):
            raise RuntimeError(
                f"Target labels are out of range. "
                f"min={min(self.targets)}, max={max(self.targets)}, "
                f"num_classes={len(self.classes)}"
            )

        print(
            f"Loaded StanfordCarsZhouDataset: split={split}, "
            f"samples={len(self.samples)}, classes={len(self.classes)}"
        )

        print("First 5 Stanford Cars samples:")
        for i in range(min(5, len(self.samples))):
            path, y = self.samples[i]
            print(
                i,
                os.path.basename(path),
                "target =",
                y,
                "class =",
                self.classes[y],
            )

    def _candidate_roots(self, split):
        train_roots = [
            os.path.join(self.dataset_root, "cars_train", "cars_train"),
            os.path.join(self.dataset_root, "cars_train"),
        ]

        test_roots = [
            os.path.join(self.dataset_root, "cars_test", "cars_test"),
            os.path.join(self.dataset_root, "cars_test"),
        ]

        if split == "train":
            return train_roots + test_roots

        if split == "test":
            return test_roots + train_roots

        return train_roots + test_roots

    def _resolve_image_path(self, rel_path, split):
        rel_path = rel_path.replace("\\", "/").lstrip("./")
        base_name = os.path.basename(rel_path)

        candidates = []

        # Case 1:
        # JSON already contains cars_train/00001.jpg or cars_test/00001.jpg
        candidates.append(os.path.join(self.dataset_root, rel_path))

        # Case 2:
        # JSON contains cars_train/00001.jpg but actual folder is
        # cars_train/cars_train/00001.jpg
        parts = rel_path.split("/")
        if len(parts) >= 2 and parts[0] in ["cars_train", "cars_test"]:
            candidates.append(
                os.path.join(self.dataset_root, parts[0], parts[0], *parts[1:])
            )

        # Case 3:
        # JSON contains only 00001.jpg.
        # Use split preferred folders first to avoid train/test filename collision.
        for root_dir in self._candidate_roots(split):
            candidates.append(os.path.join(root_dir, rel_path))
            candidates.append(os.path.join(root_dir, base_name))

        # Remove duplicates while preserving order
        candidates = list(dict.fromkeys(candidates))

        for path in candidates:
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            "Could not resolve Stanford Cars image path.\n"
            f"Relative path from JSON: {rel_path}\n"
            f"Dataset root: {self.dataset_root}\n"
            f"Checked first candidates:\n" +
            "\n".join(candidates[:10])
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target



class FGVCAircraftImageDataset(Dataset):
    """
    Custom FGVC Aircraft dataset using official split files.

    Expected structure:
        fgvc-aircraft-2013b/
            data/
                images/
                images_variant_train.txt
                images_variant_test.txt
                images_variant_val.txt
                images_variant_trainval.txt
                variants.txt
    """

    def __init__(self, dataset_root, split="test", transform=None, target_level="variant"):
        self.dataset_root = dataset_root
        self.data_root = os.path.join(dataset_root, "data")
        self.image_root = os.path.join(self.data_root, "images")
        self.transform = transform
        self.target_level = target_level

        if target_level not in ["variant", "family", "manufacturer"]:
            raise ValueError(
                f"target_level must be one of ['variant', 'family', 'manufacturer'], got {target_level}"
            )

        if split == "train":
            split_name = "train"
        elif split == "test":
            split_name = "test"
        elif split == "val":
            split_name = "val"
        elif split == "trainval":
            split_name = "trainval"
        else:
            raise ValueError(f"Unsupported split={split}")

        split_file = os.path.join(
            self.data_root,
            f"images_{target_level}_{split_name}.txt"
        )

        class_file = os.path.join(self.data_root, f"{target_level}s.txt")

        if target_level == "family":
            class_file = os.path.join(self.data_root, "families.txt")
        elif target_level == "manufacturer":
            class_file = os.path.join(self.data_root, "manufacturers.txt")
        elif target_level == "variant":
            class_file = os.path.join(self.data_root, "variants.txt")

        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        if not os.path.isfile(class_file):
            raise FileNotFoundError(f"Class file not found: {class_file}")

        with open(class_file, "r") as f:
            self.classes = [line.strip() for line in f if line.strip()]

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        self.targets = []

        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # Format is usually:
                # image_id class name with spaces
                #
                # Example:
                # 0034309 Boeing 707-320
                parts = line.split(" ", 1)

                if len(parts) != 2:
                    raise ValueError(f"Unexpected line format in {split_file}: {line}")

                image_id, class_name = parts
                image_path = os.path.join(self.image_root, image_id + ".jpg")

                if class_name not in self.class_to_idx:
                    raise KeyError(
                        f"Class '{class_name}' from split file not found in class list."
                    )

                target = self.class_to_idx[class_name]

                self.samples.append((image_path, target))
                self.targets.append(target)

        self.imgs = self.samples

        print(
            f"Loaded FGVC Aircraft: split={split}, "
            f"target_level={target_level}, samples={len(self.samples)}, "
            f"classes={len(self.classes)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    

STANFORDCARS_CLASSNAMES = [
    "AM General Hummer SUV 2000",
    "Acura RL Sedan 2012",
    "Acura TL Sedan 2012",
    "Acura TL Type-S 2008",
    "Acura TSX Sedan 2012",
    "Acura Integra Type R 2001",
    "Acura ZDX Hatchback 2012",
    "Aston Martin V8 Vantage Convertible 2012",
    "Aston Martin V8 Vantage Coupe 2012",
    "Aston Martin Virage Convertible 2012",
    "Aston Martin Virage Coupe 2012",
    "Audi RS 4 Convertible 2008",
    "Audi A5 Coupe 2012",
    "Audi TTS Coupe 2012",
    "Audi R8 Coupe 2012",
    "Audi V8 Sedan 1994",
    "Audi 100 Sedan 1994",
    "Audi 100 Wagon 1994",
    "Audi TT Hatchback 2011",
    "Audi S6 Sedan 2011",
    "Audi S5 Convertible 2012",
    "Audi S5 Coupe 2012",
    "Audi S4 Sedan 2012",
    "Audi S4 Sedan 2007",
    "Audi TT RS Coupe 2012",
    "BMW ActiveHybrid 5 Sedan 2012",
    "BMW 1 Series Convertible 2012",
    "BMW 1 Series Coupe 2012",
    "BMW 3 Series Sedan 2012",
    "BMW 3 Series Wagon 2012",
    "BMW 6 Series Convertible 2007",
    "BMW X5 SUV 2007",
    "BMW X6 SUV 2012",
    "BMW M3 Coupe 2012",
    "BMW M5 Sedan 2010",
    "BMW M6 Convertible 2010",
    "BMW X3 SUV 2012",
    "BMW Z4 Convertible 2012",
    "Bentley Continental Supersports Conv. Convertible 2012",
    "Bentley Arnage Sedan 2009",
    "Bentley Mulsanne Sedan 2011",
    "Bentley Continental GT Coupe 2012",
    "Bentley Continental GT Coupe 2007",
    "Bentley Continental Flying Spur Sedan 2007",
    "Bugatti Veyron 16.4 Convertible 2009",
    "Bugatti Veyron 16.4 Coupe 2009",
    "Buick Regal GS 2012",
    "Buick Rainier SUV 2007",
    "Buick Verano Sedan 2012",
    "Buick Enclave SUV 2012",
    "Cadillac CTS-V Sedan 2012",
    "Cadillac SRX SUV 2012",
    "Cadillac Escalade EXT Crew Cab 2007",
    "Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
    "Chevrolet Corvette Convertible 2012",
    "Chevrolet Corvette ZR1 2012",
    "Chevrolet Corvette Ron Fellows Edition Z06 2007",
    "Chevrolet Traverse SUV 2012",
    "Chevrolet Camaro Convertible 2012",
    "Chevrolet HHR SS 2010",
    "Chevrolet Impala Sedan 2007",
    "Chevrolet Tahoe Hybrid SUV 2012",
    "Chevrolet Sonic Sedan 2012",
    "Chevrolet Express Cargo Van 2007",
    "Chevrolet Avalanche Crew Cab 2012",
    "Chevrolet Cobalt SS 2010",
    "Chevrolet Malibu Hybrid Sedan 2010",
    "Chevrolet TrailBlazer SS 2009",
    "Chevrolet Silverado 2500HD Regular Cab 2012",
    "Chevrolet Silverado 1500 Classic Extended Cab 2007",
    "Chevrolet Express Van 2007",
    "Chevrolet Monte Carlo Coupe 2007",
    "Chevrolet Malibu Sedan 2007",
    "Chevrolet Silverado 1500 Extended Cab 2012",
    "Chevrolet Silverado 1500 Regular Cab 2012",
    "Chrysler Aspen SUV 2009",
    "Chrysler Sebring Convertible 2010",
    "Chrysler Town and Country Minivan 2012",
    "Chrysler 300 SRT-8 2010",
    "Chrysler Crossfire Convertible 2008",
    "Chrysler PT Cruiser Convertible 2008",
    "Daewoo Nubira Wagon 2002",
    "Dodge Caliber Wagon 2012",
    "Dodge Caliber Wagon 2007",
    "Dodge Caravan Minivan 1997",
    "Dodge Ram Pickup 3500 Crew Cab 2010",
    "Dodge Ram Pickup 3500 Quad Cab 2009",
    "Dodge Sprinter Cargo Van 2009",
    "Dodge Journey SUV 2012",
    "Dodge Dakota Crew Cab 2010",
    "Dodge Dakota Club Cab 2007",
    "Dodge Magnum Wagon 2008",
    "Dodge Challenger SRT8 2011",
    "Dodge Durango SUV 2012",
    "Dodge Durango SUV 2007",
    "Dodge Charger Sedan 2012",
    "Dodge Charger SRT-8 2009",
    "Eagle Talon Hatchback 1998",
    "FIAT 500 Abarth 2012",
    "FIAT 500 Convertible 2012",
    "Ferrari FF Coupe 2012",
    "Ferrari California Convertible 2012",
    "Ferrari 458 Italia Convertible 2012",
    "Ferrari 458 Italia Coupe 2012",
    "Fisker Karma Sedan 2012",
    "Ford F-450 Super Duty Crew Cab 2012",
    "Ford Mustang Convertible 2007",
    "Ford Freestar Minivan 2007",
    "Ford Expedition EL SUV 2009",
    "Ford Edge SUV 2012",
    "Ford Ranger SuperCab 2011",
    "Ford GT Coupe 2006",
    "Ford F-150 Regular Cab 2012",
    "Ford F-150 Regular Cab 2007",
    "Ford Focus Sedan 2007",
    "Ford E-Series Wagon Van 2012",
    "Ford Fiesta Sedan 2012",
    "GMC Terrain SUV 2012",
    "GMC Savana Van 2012",
    "GMC Yukon Hybrid SUV 2012",
    "GMC Acadia SUV 2012",
    "GMC Canyon Extended Cab 2012",
    "Geo Metro Convertible 1993",
    "HUMMER H3T Crew Cab 2010",
    "HUMMER H2 SUT Crew Cab 2009",
    "Honda Odyssey Minivan 2012",
    "Honda Odyssey Minivan 2007",
    "Honda Accord Coupe 2012",
    "Honda Accord Sedan 2012",
    "Hyundai Veloster Hatchback 2012",
    "Hyundai Santa Fe SUV 2012",
    "Hyundai Tucson SUV 2012",
    "Hyundai Veracruz SUV 2012",
    "Hyundai Sonata Hybrid Sedan 2012",
    "Hyundai Elantra Sedan 2007",
    "Hyundai Accent Sedan 2012",
    "Hyundai Genesis Sedan 2012",
    "Hyundai Sonata Sedan 2012",
    "Hyundai Elantra Touring Hatchback 2012",
    "Hyundai Azera Sedan 2012",
    "Infiniti G Coupe IPL 2012",
    "Infiniti QX56 SUV 2011",
    "Isuzu Ascender SUV 2008",
    "Jaguar XK XKR 2012",
    "Jeep Patriot SUV 2012",
    "Jeep Wrangler SUV 2012",
    "Jeep Liberty SUV 2012",
    "Jeep Grand Cherokee SUV 2012",
    "Jeep Compass SUV 2012",
    "Lamborghini Reventon Coupe 2008",
    "Lamborghini Aventador Coupe 2012",
    "Lamborghini Gallardo LP 570-4 Superleggera 2012",
    "Lamborghini Diablo Coupe 2001",
    "Land Rover Range Rover SUV 2012",
    "Land Rover LR2 SUV 2012",
    "Lincoln Town Car Sedan 2011",
    "MINI Cooper Roadster Convertible 2012",
    "Maybach Landaulet Convertible 2012",
    "Mazda Tribute SUV 2011",
    "McLaren MP4-12C Coupe 2012",
    "Mercedes-Benz 300-Class Convertible 1993",
    "Mercedes-Benz C-Class Sedan 2012",
    "Mercedes-Benz SL-Class Coupe 2009",
    "Mercedes-Benz E-Class Sedan 2012",
    "Mercedes-Benz S-Class Sedan 2012",
    "Mercedes-Benz Sprinter Van 2012",
    "Mitsubishi Lancer Sedan 2012",
    "Nissan Leaf Hatchback 2012",
    "Nissan NV Passenger Van 2012",
    "Nissan Juke Hatchback 2012",
    "Nissan 240SX Coupe 1998",
    "Plymouth Neon Coupe 1999",
    "Porsche Panamera Sedan 2012",
    "Ram C/V Cargo Van Minivan 2012",
    "Rolls-Royce Phantom Drophead Coupe Convertible 2012",
    "Rolls-Royce Ghost Sedan 2012",
    "Rolls-Royce Phantom Sedan 2012",
    "Scion xD Hatchback 2012",
    "Spyker C8 Convertible 2009",
    "Spyker C8 Coupe 2009",
    "Suzuki Aerio Sedan 2007",
    "Suzuki Kizashi Sedan 2012",
    "Suzuki SX4 Hatchback 2012",
    "Suzuki SX4 Sedan 2012",
    "Tesla Model S Sedan 2012",
    "Toyota Sequoia SUV 2012",
    "Toyota Camry Sedan 2012",
    "Toyota Corolla Sedan 2012",
    "Toyota 4Runner SUV 2012",
    "Volkswagen Golf Hatchback 2012",
    "Volkswagen Golf Hatchback 1991",
    "Volkswagen Beetle Hatchback 2012",
    "Volvo C30 Hatchback 2012",
    "Volvo 240 Sedan 1993",
    "Volvo XC90 SUV 2007",
    "smart fortwo Convertible 2012",
]

FLOWERS102_CLASSNAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]
class Flowers102MatDataset(Dataset):
    """
    Oxford Flowers 102 dataset using:
        - imagelabels.mat
        - setid.mat
        - jpg/*.jpg

    Expected structure:
        flower102/
            jpg/
                image_00001.jpg
                image_00002.jpg
                ...
            imagelabels.mat
            setid.mat
    """

    def __init__(self, dataset_root, split="test", transform=None):
        self.dataset_root = dataset_root
        self.image_root = os.path.join(dataset_root, "jpg")
        self.transform = transform

        labels_mat = os.path.join(dataset_root, "imagelabels.mat")
        split_mat = os.path.join(dataset_root, "setid.mat")

        if not os.path.isfile(labels_mat):
            raise FileNotFoundError(f"Labels file not found: {labels_mat}")
        if not os.path.isfile(split_mat):
            raise FileNotFoundError(f"Split file not found: {split_mat}")
        if not os.path.isdir(self.image_root):
            raise FileNotFoundError(f"Image folder not found: {self.image_root}")

        labels_data = loadmat(labels_mat)
        split_data = loadmat(split_mat)

        # imagelabels.mat usually stores 1-based labels of shape (1, N)
        labels = labels_data["labels"].squeeze()

        if split == "train":
            ids = split_data["trnid"].squeeze()
        elif split == "test":
            ids = split_data["tstid"].squeeze()
        elif split in ["val", "valid", "validation"]:
            ids = split_data["valid"].squeeze()
        else:
            raise ValueError("split must be one of: train, test, val")

        # Convert to Python ints
        ids = [int(x) for x in ids.tolist()]

        # Flowers labels are 1-based in the .mat file. Convert to 0-based.
        self.samples = []
        self.targets = []

        for img_id in ids:
            image_name = f"image_{img_id:05d}.jpg"
            image_path = os.path.join(self.image_root, image_name)

            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            target = int(labels[img_id - 1]) - 1

            self.samples.append((image_path, target))
            self.targets.append(target)

        self.imgs = self.samples

        self.classes = FLOWERS102_CLASSNAMES
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        print(
            f"Loaded Flowers102MatDataset: split={split}, "
            f"samples={len(self.samples)}, classes={len(self.classes)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target

class EuroSATCSVSplitDataset(Dataset):
    """
    EuroSAT loader using train.csv, test.csv, validation.csv and label_map.json.

    Expected structure:
        EuroSAT/
            AnnualCrop/
            Forest/
            HerbaceousVegetation/
            Highway/
            Industrial/
            Pasture/
            PermanentCrop/
            Residential/
            River/
            SeaLake/
            label_map.json
            train.csv
            test.csv
            validation.csv
    """

    def __init__(self, dataset_root, split="test", transform=None):
        self.dataset_root = os.path.abspath(dataset_root)
        self.split = split
        self.transform = transform

        label_map_path = os.path.join(self.dataset_root, "label_map.json")

        if not os.path.isfile(label_map_path):
            raise FileNotFoundError(f"label_map.json not found:\n{label_map_path}")

        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        # label_map format:
        # {"AnnualCrop": 0, "Forest": 1, ...}
        self.class_to_idx = {
            str(class_name): int(idx)
            for class_name, idx in label_map.items()
        }

        self.classes = [
            class_name
            for class_name, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1])
        ]

        if split == "train":
            csv_name = "train.csv"
        elif split == "test":
            csv_name = "test.csv"
        elif split in ["val", "valid", "validation"]:
            csv_name = "validation.csv"
        else:
            raise ValueError(
                f"Unsupported EuroSAT split={split}. "
                "Use train, test, val, valid, or validation."
            )

        csv_path = os.path.join(self.dataset_root, csv_name)

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"EuroSAT split CSV not found:\n{csv_path}")

        self.samples = []
        self.targets = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) == 0:
            raise RuntimeError(f"CSV file is empty:\n{csv_path}")

        header = [h.strip() for h in rows[0]]
        has_header = any(
            h.lower() in [
                "path", "filepath", "file_path", "filename", "image", "image_path",
                "label", "class", "class_name", "target"
            ]
            for h in header
        )

        if has_header:
            data_rows = rows[1:]
            col_to_idx = {name.strip().lower(): i for i, name in enumerate(header)}

            path_col = None
            for key in ["path", "filepath", "file_path", "filename", "image", "image_path"]:
                if key in col_to_idx:
                    path_col = col_to_idx[key]
                    break

            label_col = None
            for key in ["label", "class", "class_name", "target"]:
                if key in col_to_idx:
                    label_col = col_to_idx[key]
                    break

            if path_col is None:
                raise ValueError(
                    f"Could not identify image path column in:\n{csv_path}\n"
                    f"Header: {header}"
                )

        else:
            data_rows = rows
            path_col = 0
            label_col = 1 if len(rows[0]) > 1 else None

        for row in data_rows:
            if len(row) == 0:
                continue

            rel_path = row[path_col].strip().replace("\\", "/").lstrip("./")

            if label_col is not None and label_col < len(row):
                raw_label = row[label_col].strip()
                target = self._parse_label(raw_label)
            else:
                # Infer label from folder name in path, e.g. AnnualCrop/image.jpg
                class_name = rel_path.split("/")[0]
                if class_name not in self.class_to_idx:
                    raise ValueError(
                        f"Could not infer EuroSAT class from path: {rel_path}"
                    )
                target = self.class_to_idx[class_name]

            image_path = self._resolve_image_path(rel_path, target)

            self.samples.append((image_path, target))
            self.targets.append(target)

        self.imgs = self.samples

        if len(self.samples) == 0:
            raise RuntimeError(f"No EuroSAT samples loaded from:\n{csv_path}")

        print(
            f"Loaded EuroSATCSVSplitDataset: split={split}, "
            f"samples={len(self.samples)}, classes={len(self.classes)}"
        )

        print("First 5 EuroSAT samples:")
        for i in range(min(5, len(self.samples))):
            path, y = self.samples[i]
            print(
                i,
                os.path.basename(path),
                "target =",
                y,
                "class =",
                self.classes[y],
            )

    def _parse_label(self, raw_label):
        raw_label = str(raw_label).strip()

        if raw_label in self.class_to_idx:
            return self.class_to_idx[raw_label]

        try:
            target = int(raw_label)
            if target < 0 or target >= len(self.classes):
                raise ValueError
            return target
        except Exception:
            raise ValueError(
                f"Could not parse EuroSAT label: {raw_label}. "
                f"Expected class name or integer label."
            )

    def _resolve_image_path(self, rel_path, target):
        base_name = os.path.basename(rel_path)
        class_name = self.classes[target]

        candidates = [
            os.path.join(self.dataset_root, rel_path),
            os.path.join(self.dataset_root, class_name, base_name),
        ]

        # If CSV stores only filename, search inside the labeled class folder.
        if "/" not in rel_path:
            candidates.append(os.path.join(self.dataset_root, class_name, rel_path))

        candidates = list(dict.fromkeys(candidates))

        for path in candidates:
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            "Could not resolve EuroSAT image path.\n"
            f"Relative path from CSV: {rel_path}\n"
            f"Target class: {target} ({class_name})\n"
            f"Checked candidates:\n" +
            "\n".join(candidates)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


# CLIP_TRANSFORM = transforms.Compose([
#     transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.CenterCrop(224),
#     transforms.Lambda(lambda img: img.convert("RGB")),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.48145466, 0.4578275, 0.40821073],
#         std=[0.26862954, 0.26130258, 0.27577711],
#     ),
# ])


SIGLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

CLIP_TRANSFORM = SIGLIP_TRANSFORM



DATASET_INFO = {

    "multimodal": {
        "dataset_dir": "multimodal",
        "loader": "multimodal_disaster",
    },
    "caltech101": {
        "dataset_dir": "caltech-101",
        "image_dir": "101_ObjectCategories",
        "loader": "json_imagefolder",
    },
    "caltech-101": {
        "dataset_dir": "caltech-101",
        "image_dir": "101_ObjectCategories",
        "loader": "json_imagefolder",
    },

    "dtd": {
        "dataset_dir": "dtd",
        "image_dir": "images",
        "loader": "json_imagefolder",
    },

    "eurosat": {
        "dataset_dir": "EuroSAT",
        "loader": "eurosat_csv",
    },
    "eurosat-rgb": {
        "dataset_dir": "EuroSAT",
        "loader": "eurosat_csv",
    },

    "food101": {
        "dataset_dir": "food-101",
        "image_dir": "food-101/images",
        "loader": "json_flat",
    },
    "food-101": {
        "dataset_dir": "food-101",
        "image_dir": "food-101/images",
        "loader": "json_flat",
    },

    "flowers102": {
    "dataset_dir": "flower102",
    "loader": "flowers_mat",
    },
    "flower102": {
        "dataset_dir": "flower102",
        "loader": "flowers_mat",
    },

    "oxfordpets": {
        "dataset_dir": "oxford-iiit-pet",
        "image_dir": "images",
        "loader": "json_flat",
    },
    "oxford-iiit-pet": {
        "dataset_dir": "oxford-iiit-pet",
        "image_dir": "images",
        "loader": "json_flat",
    },

    "stanfordcars": {
        "dataset_dir": "stanford-cars",
        "image_dirs": [
            "cars_train/cars_train",
            "cars_test/cars_test",
        ],
        "loader": "json_flat",
    },
    "stanfordcars": {
    "dataset_dir": "stanford-cars",
    "image_dirs": [
        "cars_train/cars_train",
        "cars_test/cars_test",
    ],
    "loader": "json_flat",
    "class_names": STANFORDCARS_CLASSNAMES,
    },

    "ucf101": {
        "dataset_dir": "UCF-101-midframes",
        "image_dir": "",
        "loader": "json_imagefolder",
    },
    "ucf-101": {
        "dataset_dir": "UCF-101-midframes",
        "image_dir": "",
        "loader": "json_imagefolder",
    },
}


def _find_json_split_file(dataset_root):
    for fname in os.listdir(dataset_root):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(dataset_root, fname)

        try:
            with open(path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict) and "train" in data and "test" in data:
                return path
        except Exception:
            pass

    raise FileNotFoundError(
        f"No JSON split file with 'train' and 'test' found in {dataset_root}"
    )


def _read_json_split_paths(dataset_root, split):
    """
    Reads JSON split file and returns image paths for the requested split.

    Expected format:
        {
            "train": [
                ["apple_pie/3113710.jpg", 0, "apple_pie"],
                ...
            ],
            "test": [
                ["apple_pie/1214326.jpg", 0, "apple_pie"],
                ...
            ]
        }
    """
    split_file = _find_json_split_file(dataset_root)

    with open(split_file, "r") as f:
        data = json.load(f)

    if split not in data:
        raise KeyError(
            f"Split '{split}' not found in JSON split file:\n{split_file}\n"
            f"Available keys: {list(data.keys())}"
        )

    split_paths = set()

    for item in data[split]:
        # item example:
        # ["apple_pie/3113710.jpg", 0, "apple_pie"]
        if isinstance(item, (list, tuple)):
            img_path = item[0]
        elif isinstance(item, str):
            img_path = item
        else:
            raise TypeError(f"Unsupported split entry type: {type(item)}")

        img_path = str(img_path).replace("\\", "/").lstrip("./")
        split_paths.add(img_path)

    return split_paths, split_file


def _imagefolder_with_json_split(dataset_root, image_root, split, transform):
    split_paths, split_file = _read_json_split_paths(dataset_root, split)

    full_dataset = datasets.ImageFolder(
        root=image_root,
        transform=transform,
    )

    image_root_path = Path(image_root).resolve()
    indices = []

    for idx, (img_path, target) in enumerate(full_dataset.samples):
        img_path_obj = Path(img_path).resolve()
        rel_path = img_path_obj.relative_to(image_root_path).as_posix()
        file_name = img_path_obj.name

        matched = (
            rel_path in split_paths
            or file_name in split_paths
            or any(rel_path.endswith(p) for p in split_paths)
        )

        if matched:
            indices.append(idx)

    if len(indices) == 0:
        raise RuntimeError(
            f"No images matched the JSON split.\n\n"
            f"Dataset root: {dataset_root}\n"
            f"Image root: {image_root}\n"
            f"Split file: {split_file}\n"
            f"Requested split: {split}\n"
            f"ImageFolder samples: {len(full_dataset)}\n"
            f"JSON split entries: {len(split_paths)}\n\n"
            f"Example ImageFolder path:\n"
            f"{full_dataset.samples[0][0] if len(full_dataset.samples) > 0 else 'NO IMAGEFOLDER SAMPLES'}\n\n"
            f"Example JSON split path:\n"
            f"{next(iter(split_paths)) if len(split_paths) > 0 else 'NO JSON SPLIT ENTRIES'}"
        )

    subset = Subset(full_dataset, indices)

    subset.classes = full_dataset.classes
    subset.class_to_idx = full_dataset.class_to_idx
    subset.samples = [full_dataset.samples[i] for i in indices]
    subset.targets = [full_dataset.targets[i] for i in indices]
    subset.imgs = subset.samples

    print(
        f"Loaded {len(subset)} samples from split='{split}' "
        f"using split file: {split_file}"
    )

    return subset



class MultimodalDisasterDataset(Dataset):
    """
    Loader for:

        multimodal/
            damaged_infrastructure/
                images/
                text/
            damaged_nature/
                images/
                text/
            fires/
                images/
                text/
            flood/
                images/
                text/
            human_damage/
                images/
                text/
            non_damage/
                images/
                text/

    Returns:
        image, target

    The text folders are preserved in self.text_paths but are not returned,
    so the dataset remains compatible with the rest of your CLIP pipeline.
    """

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    def __init__(
        self,
        dataset_root,
        split="test",
        transform=None,
        train_ratio=0.8,
        seed=42,
    ):
        self.dataset_root = os.path.abspath(dataset_root)
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.seed = seed

        if split not in ["train", "test"]:
            raise ValueError(f"split must be 'train' or 'test', got: {split}")

        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"Multimodal dataset folder not found:\n{self.dataset_root}")

        class_dirs = [
            d for d in sorted(os.listdir(self.dataset_root))
            if os.path.isdir(os.path.join(self.dataset_root, d))
        ]

        if len(class_dirs) == 0:
            raise RuntimeError(f"No class folders found in:\n{self.dataset_root}")

        self.original_classes = class_dirs
        self.classes = [normalize_class_name(c) for c in class_dirs]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        all_class_samples = []

        for target, folder_name in enumerate(class_dirs):
            class_root = os.path.join(self.dataset_root, folder_name)
            image_root = os.path.join(class_root, "images")
            text_root = os.path.join(class_root, "text")

            if not os.path.isdir(image_root):
                raise FileNotFoundError(
                    f"Expected images folder not found for class '{folder_name}':\n"
                    f"{image_root}"
                )

            image_paths = []

            for ext in self.IMG_EXTENSIONS:
                image_paths.extend(
                    glob.glob(os.path.join(image_root, "**", f"*{ext}"), recursive=True)
                )
                image_paths.extend(
                    glob.glob(os.path.join(image_root, "**", f"*{ext.upper()}"), recursive=True)
                )

            image_paths = sorted(list(dict.fromkeys(image_paths)))

            if len(image_paths) == 0:
                print(f"Warning: no images found for class '{folder_name}' in {image_root}")
                continue

            class_samples = []

            bad_count = 0

            for image_path in image_paths:
                if not self._is_valid_image(image_path):
                    bad_count += 1
                    print(f"Skipping corrupted image: {image_path}")
                    continue

                text_path = self._find_matching_text(image_path, text_root)
                class_samples.append((image_path, target, text_path))

            if bad_count > 0:
                print(f"Skipped {bad_count} corrupted images from class '{folder_name}'")

            all_class_samples.append(class_samples)

        self.samples = []
        self.targets = []
        self.text_paths = []

        for target, class_samples in enumerate(all_class_samples):
            selected = self._split_class_samples(class_samples, split)

            for image_path, y, text_path in selected:
                self.samples.append((image_path, y))
                self.targets.append(y)
                self.text_paths.append(text_path)

        self.imgs = self.samples

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples loaded for split='{split}' from:\n{self.dataset_root}"
            )

        print(
            f"Loaded MultimodalDisasterDataset: split={split}, "
            f"samples={len(self.samples)}, classes={len(self.classes)}"
        )

        print("Classes:")
        for i, cls_name in enumerate(self.classes):
            print(f"  {i}: {cls_name}")

    def _split_class_samples(self, class_samples, split):
        """
        Deterministic per-class split.
        Since your folder does not show explicit train/test folders,
        this creates an 80/20 stratified split by default.
        """

        n = len(class_samples)

        if n == 1:
            return class_samples

        import random
        rng = random.Random(self.seed)
        shuffled = class_samples[:]
        rng.shuffle(shuffled)

        n_train = int(round(n * self.train_ratio))
        n_train = max(1, min(n_train, n - 1))

        if split == "train":
            return shuffled[:n_train]

        return shuffled[n_train:]

    def _find_matching_text(self, image_path, text_root):
        if not os.path.isdir(text_root):
            return None

        stem = os.path.splitext(os.path.basename(image_path))[0]

        candidates = [
            os.path.join(text_root, stem + ".txt"),
            os.path.join(text_root, stem + ".json"),
            os.path.join(text_root, stem + ".csv"),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

        return None
    def _is_valid_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()

            # Reopen after verify because verify() invalidates the image object.
            with Image.open(image_path) as img:
                img.convert("RGB")

            return True

        except Exception:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise OSError(f"Failed to read image: {image_path}") from e

        if self.transform is not None:
            image = self.transform(image)

        return image, target




def create_torchvision_dataset(
    dataset_name: str,
    root: str,
    split: str = "test",
    download: bool = True,
):
    """
    Dataset factory.

    CIFAR10/CIFAR100:
        Uses torchvision datasets.

    FGVC Aircraft:
        Uses custom official txt split loader.

    Oxford Pets / Stanford Cars:
        Uses JSON split loader because images are not arranged as ImageFolder.

    Caltech101 / DTD / EuroSAT / Food101 / UCF101:
        Uses ImageFolder plus JSON train/test split filtering.
    """

    name = dataset_name.lower().replace("_", "-")

    if split not in ["train", "test"]:
        raise ValueError(f"split must be 'train' or 'test', got: {split}")

    # -------------------------------------------------
    # CIFAR datasets
    # -------------------------------------------------
    if name in ["cifar10", "cifar-10"]:
        return datasets.CIFAR10(
            root=root,
            train=(split == "train"),
            transform=CLIP_TRANSFORM,
            download=download,
        )

    if name in ["cifar100", "cifar-100"]:
        return datasets.CIFAR100(
            root=root,
            train=(split == "train"),
            transform=CLIP_TRANSFORM,
            download=download,
        )

    # -------------------------------------------------
    # FGVC Aircraft
    # Not ImageFolder. Uses official txt split files.
    # -------------------------------------------------
    if name in ["fgvcaircraft", "fgvc-aircraft"]:
        dataset_root = os.path.join(root, "fgvc-aircraft-2013b")

        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(
                f"FGVC Aircraft folder not found:\n{dataset_root}"
            )

        return FGVCAircraftImageDataset(
            dataset_root=dataset_root,
            split=split,
            transform=CLIP_TRANSFORM,
            target_level="variant",
        )
    

    # -------------------------------------------------
    # Remaining datasets
    # -------------------------------------------------
    if name not in DATASET_INFO:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name}.\n"
            f"Supported names are:\n"
            f"{list(DATASET_INFO.keys()) + ['cifar10', 'cifar100', 'fgvcaircraft', 'multimodal']}"
        )

    info = DATASET_INFO[name]

    dataset_root = os.path.join(root, info["dataset_dir"])

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset folder not found:\n{dataset_root}\n\n"
            f"Check root and dataset_name. Example:\n"
            f"dataset_name='food101', root='./data'"
        )
    if info["loader"] == "multimodal_disaster":
        return MultimodalDisasterDataset(
            dataset_root=dataset_root,
            split=split,
            transform=CLIP_TRANSFORM,
            train_ratio=0.8,
            seed=42,
        )

    # -------------------------------------------------
    # Flat image datasets:
    # Oxford Pets:
    #   oxford-iiit-pet/images/*.jpg
    #
    # Stanford Cars:
    #   stanford_cars/cars_train/cars_train/*.jpg
    #   stanford_cars/cars_test/cars_test/*.jpg
    # -------------------------------------------------
    if info["loader"] == "json_flat":
        if "image_dirs" in info:
            image_roots = [
                os.path.join(dataset_root, image_dir)
                for image_dir in info["image_dirs"]
            ]
        else:
            image_roots = os.path.join(dataset_root, info["image_dir"])

        return JSONSplitImageDataset(
            dataset_root=dataset_root,
            image_roots=image_roots,
            split=split,
            transform=CLIP_TRANSFORM,
        )
    if info["loader"] == "flowers_mat":
        return Flowers102MatDataset(
            dataset_root=dataset_root,
            split=split,
            transform=CLIP_TRANSFORM,
        )
    # -------------------------------------------------
    # ImageFolder datasets with JSON split:
    # dataset_root contains split_zhou_*.json
    # image_root contains class folders
    # -------------------------------------------------
    if info["loader"] == "json_imagefolder":
        image_root = os.path.join(dataset_root, info["image_dir"])

        if not os.path.isdir(image_root):
            raise FileNotFoundError(
                f"Image root folder not found:\n{image_root}\n\n"
                f"dataset_root was:\n{dataset_root}"
            )

        return _imagefolder_with_json_split(
            dataset_root=dataset_root,
            image_root=image_root,
            split=split,
            transform=CLIP_TRANSFORM,
        )
    
        # -------------------------------------------------
    # Stanford Cars
    # Uses split_zhou_StanfordCars.json as the source of truth.
    # This avoids incorrect hardcoded class ordering.
    # -------------------------------------------------
    if name in ["stanfordcars", "stanford-cars"]:
        dataset_root = os.path.join(root, "stanford-cars")

        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(
                f"Stanford Cars folder not found:\n{dataset_root}\n\n"
                f"Expected structure:\n"
                f"{root}/stanford-cars/\n"
                f"    car_devkit/\n"
                f"    cars_train/\n"
                f"    cars_test/\n"
                f"    split_zhou_StanfordCars.json"
            )

        return StanfordCarsZhouDataset(
            dataset_root=dataset_root,
            split=split,
            transform=CLIP_TRANSFORM,
        )
    if info["loader"] == "eurosat_csv":
        dataset_root = os.path.join(root, info["dataset_dir"])

        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(
                f"EuroSAT folder not found:\n{dataset_root}"
            )

        return EuroSATCSVSplitDataset(
            dataset_root=dataset_root,
            split=split,
            transform=CLIP_TRANSFORM,
        )

    raise ValueError(f"Unknown loader type: {info['loader']}")