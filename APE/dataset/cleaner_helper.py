import re


def description_cleaner(description: str):
    line = re.sub(r"-+", " ", description)
    line = re.sub(r"[^a-zA-Z0-9, ]+", " ", line)
    line = re.sub(r"[ ]+", " ", line)
    line += "."

    return line.strip()
