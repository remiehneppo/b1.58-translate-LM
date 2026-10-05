# scripts/clean_data.py
import json
import os
import sys
import unicodedata
from tqdm import tqdm


def is_valid_char(c: str) -> bool:
    # 1. Whitespace, numbers, common punctuation, symbols & combining marks
    cat = unicodedata.category(c)
    if cat.startswith(("Z", "N", "P", "S", "M")):
        return True

    # 2. Check script via Unicode character name
    name = unicodedata.name(c, "")
    if "LATIN" in name or "CYRILLIC" in name:
        return True

    return False


def is_valid_text(text: str) -> bool:
    norm_text = unicodedata.normalize("NFC", text)
    return bool(norm_text) and all(is_valid_char(c) for c in norm_text)


def clean_jsonl(input_path: str, output_path: str = None):
    if output_path is None:
        output_path = input_path

    temp_path = input_path + ".tmp"
    valid_count = 0
    removed_count = 0
    removed_reasons = []

    with open(input_path, "r", encoding="utf-8") as f, open(temp_path, "w", encoding="utf-8") as out:
        for idx, line in enumerate(tqdm(f, desc="Cleaning dataset")):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
            except (json.JSONDecodeError, Exception):
                removed_count += 1
                continue

            src = (item.get("src") or "").strip()
            tgt = (item.get("tgt") or "").strip()
            if is_valid_text(src) and is_valid_text(tgt):
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                valid_count += 1
            else:
                removed_count += 1
                if len(removed_reasons) < 50:
                    bad_chars = [c for c in src + tgt if not is_valid_char(c)]
                    removed_reasons.append((idx + 1, bad_chars[:3]))

    os.replace(temp_path, output_path)
    print(f"\n[Done] Kept: {valid_count} samples | Removed: {removed_count} samples.")
    if removed_reasons:
        print("Sample removed items (line, bad_chars):", removed_reasons[:5])


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "distill_vi_ru_tatoeba.jsonl"
    clean_jsonl(target_file)
