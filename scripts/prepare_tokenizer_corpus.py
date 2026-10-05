# scripts/prepare_tokenizer_corpus.py
import json
import sys
from tqdm import tqdm


def extract_corpus(input_jsonl: str, output_txt: str = "corpus_tokenizer.txt"):
    unique_sentences = set()

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Extracting text"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, Exception):
                continue
            src = (item.get("src") or "").strip().replace("\r", " ").replace("\n", " ")
            tgt = (item.get("tgt") or "").strip().replace("\r", " ").replace("\n", " ")
            if src:
                unique_sentences.add(src)
            if tgt:
                unique_sentences.add(tgt)

    with open(output_txt, "w", encoding="utf-8") as out:
        for sent in unique_sentences:
            out.write(sent + "\n")

    print(f"[Done] Exported {len(unique_sentences)} unique sentences to '{output_txt}'.")


if __name__ == "__main__":
    src_file = sys.argv[1] if len(sys.argv) > 1 else "distill_vi_ru_tatoeba.jsonl"
    dest_file = sys.argv[2] if len(sys.argv) > 2 else "corpus_tokenizer.txt"
    extract_corpus(src_file, dest_file)
