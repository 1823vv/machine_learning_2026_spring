from pathlib import Path
import sys


def get_ignore_list():
    """Read convert_ignore.txt and return set of filenames to ignore."""
    ignore_file = Path.cwd() / "convert_ignore.txt"
    if ignore_file.exists():
        with ignore_file.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def convert_md_to_ipynb():
    try:
        import notedown
        import nbformat
    except ImportError:
        print("Missing required package: notedown")
        print("Install with: pip install notedown")
        sys.exit(1)

    ignore_list = get_ignore_list()
    cwd = Path.cwd()
    md_files = list(cwd.glob("*.md"))

    if not md_files:
        print("No .md files found in current directory.")
        return

    for md_file in md_files:
        # Skip if file is in ignore list
        if md_file.name in ignore_list:
            print(f"Skipping: {md_file.name} (in convert_ignore.txt)")
            continue

        ipynb_file = md_file.with_suffix(".ipynb")

        # Skip if notebook already exists
        if ipynb_file.exists():
            print(f"Skipping: {md_file.name} -> {ipynb_file.name} already exists")
            continue

        try:
            print(f"Converting: {md_file.name} -> {ipynb_file.name}")

            # Read markdown
            with md_file.open("r", encoding="utf-8") as f:
                reader = notedown.MarkdownReader()
                notebook = reader.read(f)

            # Write notebook
            with ipynb_file.open("w", encoding="utf-8") as f:
                nbformat.write(notebook, f)

            print(f"Done: {ipynb_file.name}")

        except Exception as e:
            print(f"Failed to convert {md_file.name}: {e}")

if __name__ == "__main__":
    convert_md_to_ipynb()
