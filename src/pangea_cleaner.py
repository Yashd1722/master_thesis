from pathlib import Path
import pandas as pd


def find_project_root() -> Path:
    """
    Finds the Master_thesis/Main folder by going one level up from this script (src -> Main).
    """
    return Path(__file__).resolve().parents[1]


def find_header_line(tab_file: Path) -> int:
    """
    The table starts right AFTER a line that is exactly '*/'.
    """
    with tab_file.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.strip() == "*/":
                return i + 1
    return 0


def main():
    root = find_project_root()  # .../Main
    input_folder = root / "dataset" / "pangaea_923197" / "datasets"
    output_folder = root / "dataset" / "pangaea_923197" / "datasets" / "clean_dataset"

    tab_files = sorted(input_folder.glob("*.tab"))

    if not tab_files:
        print("No .tab files found in:", input_folder)
        print("Tip: check with: ls -lh", input_folder)
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    for tab_file in tab_files:
        start_line = find_header_line(tab_file)

        df = pd.read_csv(tab_file, sep="\t", skiprows=start_line, engine="python")
        df = df.dropna(axis=1, how="all")

        out_csv = output_folder / (tab_file.stem + ".csv")
        df.to_csv(out_csv, index=False)

        print(f"Converted: {tab_file.name} -> {out_csv}")

    print("\nDone. Output folder:", output_folder)


if __name__ == "__main__":
    main()
