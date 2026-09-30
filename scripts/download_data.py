"""Download the two course datasets next to the notebooks (standard library only)."""

from pathlib import Path
from urllib.request import urlopen


DATASETS = {
    "names.txt": "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
    "input.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def main():
    destination = Path(__file__).resolve().parents[1] / "notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    for name, url in DATASETS.items():
        target = destination / name
        if target.exists():
            print(f"Already exists: {target}")
            continue
        print(f"Downloading {name}...")
        with urlopen(url, timeout=60) as response:
            content = response.read()
        target.write_bytes(content)
        print(f"Saved {target} ({len(content):,} bytes)")


if __name__ == "__main__":
    main()
