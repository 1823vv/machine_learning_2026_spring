from pathlib import Path
import subprocess
import sys
import os


def get_ignore_list():
    """Read convert_ignore.txt and return set of filenames to ignore."""
    ignore_file = Path.cwd() / "convert_ignore.txt"
    if ignore_file.exists():
        with ignore_file.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def convert_ipynb_to_md():
    ignore_list = get_ignore_list()
    cwd = Path.cwd()
    notebooks = list(cwd.glob("*.ipynb"))

    if not notebooks:
        print("No .ipynb files found in current directory.")
        return

    for ipynb_file in notebooks:
        # Skip if file is in ignore list
        if ipynb_file.name in ignore_list:
            print(f"Skipping: {ipynb_file.name} (in convert_ignore.txt)")
            continue

        md_file = ipynb_file.with_suffix(".md")

        # Skip existing markdown
        if md_file.exists():
            print(f"Skipping: {ipynb_file.name} -> {md_file.name} already exists")
            continue

        try:
            print(f"Converting: {ipynb_file.name} -> {md_file.name}")

            # Force exclude ALL outputs
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "markdown",
                    str(ipynb_file),
                    "--no-prompt",
                    "--ClearOutputPreprocessor.enabled=True",
                ],
                check=True,
                env={
                    **os.environ,
                    "JUPYTER_PATH": "",
                    "JUPYTER_CONFIG_DIR": "",
                },
            )

            print(f"Done: {md_file.name}")

        except subprocess.CalledProcessError:
            print(
                f"nbconvert failed for {ipynb_file.name}, trying manual conversion..."
            )

            # Fallback: pure python conversion without outputs
            try:
                import nbformat

                nb = nbformat.read(ipynb_file, as_version=4)

                lines = []

                for cell in nb.cells:
                    if cell.cell_type == "markdown":
                        lines.append(cell.source)
                        lines.append("")

                    elif cell.cell_type == "code":
                        # Only source code, NO outputs
                        lines.append("```python")
                        lines.append(cell.source)
                        lines.append("```")
                        lines.append("")

                md_file.write_text("\n".join(lines), encoding="utf-8")

                print(f"Manual conversion done: {md_file.name}")

            except Exception as e:
                print(f"Failed manual conversion for {ipynb_file.name}: {e}")


if __name__ == "__main__":
    convert_ipynb_to_md()
