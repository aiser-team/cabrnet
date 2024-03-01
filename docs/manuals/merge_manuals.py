import argparse

md_files = [
    "index.md",
    "install.md",
    "cabrnet.md",
    "applications.md",
    "model.md",
    "data.md",
    "training.md",
    "visualize.md",
    "mnist.md",
    "legacy.md",
    "download.md",
]


def build_index_table(file_list: list[str]) -> dict[str, str]:
    # Build matching between markdown files and their first section
    index_table = {}
    for filename in file_list:
        with open(filename, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                if line.startswith("#"):
                    chap_name = line.replace("#", "").strip().replace(" ", "-").lower()
                    index_table[filename] = chap_name
                    break
    return index_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Merge the following markdown files and fixes links between files {md_files}"
    )
    parser.add_argument(
        "--output", "-o", type=str, metavar="path/to/file", default="user_manual.md", help="path to output file"
    )
    args = parser.parse_args()
    replace_table = build_index_table(md_files)

    # Merge and replace
    with open(args.output, "w") as fout:
        for filename in md_files:
            with open(filename, "r") as file:
                filedata = file.read()
            for key in replace_table:
                filedata = filedata.replace(key, f"#{replace_table[key]}")
            fout.write(filedata)
