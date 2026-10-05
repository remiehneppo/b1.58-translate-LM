# scripts/generate_teacher_openai.py
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
import os
import re
from time import sleep
from typing import List
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


class TripletTranslationItem(BaseModel):
    en: str = Field(description="Original English sentence")
    vi: str = Field(description="Natural, accurate translation in Vietnamese")
    ru: str = Field(description="Natural, accurate translation in Russian")


class TripletTranslationBatch(BaseModel):
    items: List[TripletTranslationItem]


def parse_response_to_items(raw_text: str, chunk: List[str]) -> List[TripletTranslationItem]:
    # 1. Thử trích xuất JSON hoặc JSON nằm trong ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    json_candidate = match.group(1) if match else None
    if not json_candidate:
        match_brace = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if match_brace:
            json_candidate = match_brace.group(1)

    if json_candidate:
        try:
            batch = TripletTranslationBatch.model_validate_json(json_candidate)
            if batch.items:
                return batch.items
        except Exception:
            try:
                data = json.loads(json_candidate)
                items_list = data.get("items", [])
                parsed = [
                    TripletTranslationItem(
                        en=str(it.get("en") or ""),
                        vi=str(it.get("vi") or ""),
                        ru=str(it.get("ru") or ""),
                    )
                    for it in items_list
                    if isinstance(it, dict)
                ]
                if parsed:
                    return parsed
            except Exception:
                pass

    # 2. Fallback nếu model sinh markdown danh sách (1. * **Vietnamese:** ... * **Russian:** ...)
    items = []
    blocks = re.split(r"\n(?=\s*\d+[\.\)])", raw_text.strip())
    for idx, b in enumerate(blocks):
        vi_m = re.search(r"\*?\*?(?:Vietnamese|Tiếng Việt):\*?\*?\s*(.+)", b, re.IGNORECASE)
        ru_m = re.search(r"\*?\*?(?:Russian|Tiếng Nga):\*?\*?\s*(.+)", b, re.IGNORECASE)
        if vi_m and ru_m:
            en_text = chunk[idx] if idx < len(chunk) else ""
            items.append(TripletTranslationItem(
                en=en_text,
                vi=vi_m.group(1).strip().strip('"*'),
                ru=ru_m.group(1).strip().strip('"*'),
            ))
    return items


def generate_vi_ru_from_en(
    input_file: str,
    output_file: str,
    model: str = "gpt-4o-mini",
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key: str = os.getenv("OPENAI_API_KEY", ""),
    batch_size: int = 15,
    overwrite: bool = False,
    workers: int = 8,
):
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    if not api_key:
        raise ValueError("API key must be provided via --api_key or OPENAI_API_KEY environment variable")
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)

    with open(input_file, "r", encoding="utf-8") as input_f:
        lines = (text for line in input_f if (text := line.strip()))

        # Kiểm tra cờ overwrite
        if overwrite and os.path.exists(output_file):
            os.remove(output_file)
            print(f"[*] Đã xóa file cũ '{output_file}' do có cờ --overwrite.")

        # Tự động Resume: Đếm số dòng đã ghi trong output_file
        start_idx = 0
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or set(row) != {"en", "ru", "vi"}:
                        raise ValueError("Output must use en/ru/vi fields; use a new output file or --overwrite")
                    if row["en"] != next(lines, None):
                        raise ValueError("Output English sentences do not match the input order")
                    start_idx += 1
            existing_lines = start_idx
            print(f"[*] Phát hiện {existing_lines} mẫu đã có. Tự động Resume từ câu thứ {start_idx}...")

        def translate_batch(i, chunk):
            prompt = (
                "Translate each English sentence into both Vietnamese and Russian accurately.\n"
                "Return a JSON object with this format: {\"items\": [{\"en\": \"...\", \"vi\": \"...\", \"ru\": \"...\"}]}\n"
                "English sentences:\n" + json.dumps(chunk, ensure_ascii=False)
            )

            for attempt in range(11):  # Initial request plus up to 10 retries.
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a professional multi-language translator. "
                                    "Translate directly and respond with a valid JSON object only. "
                                    "Schema: {\"items\": [{\"en\": \"...\", \"vi\": \"...\", \"ru\": \"...\"}]}"
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        extra_body={
                            "reasoning_effort": "none",
                            "thinking": {"type": "disabled"},
                            "chat_template_kwargs": {"thinking": False},
                        },
                    )
                    raw_text = response.choices[0].message.content or ""
                    parsed_items = parse_response_to_items(raw_text, chunk)
                    if len(parsed_items) != len(chunk) or any(not item.vi.strip() or not item.ru.strip() for item in parsed_items):
                        raise ValueError(f"Batch at sentence {i} has incomplete translations")
                    for item, en in zip(parsed_items, chunk):
                        item.en = en
                    return parsed_items
                except Exception as error:
                    if attempt == 10:
                        raise
                    delay = (attempt + 1) * 5
                    tqdm.write(f"[Warning] Batch at sentence {i}: {error}. Retry {attempt + 1}/10 in {delay}s")
                    sleep(delay)

        total_items = start_idx
        def batches():
            i = start_idx
            while chunk := list(islice(lines, batch_size)):
                yield i, chunk
                i += len(chunk)

        batch_iter = batches()
        with ThreadPoolExecutor(max_workers=workers) as executor, open(output_file, "a", encoding="utf-8") as out_f, tqdm(
            initial=start_idx // batch_size,
            desc="Generating En-Ru-Vi Triplets",
            unit="batch",
        ) as pbar:
            pending = deque(executor.submit(translate_batch, i, chunk) for i, chunk in islice(batch_iter, workers))
            if not pending:
                print("[+] Toàn bộ file dữ liệu đã được dịch hoàn tất.")
            while pending:
                try:
                    parsed_items = pending.popleft().result()
                except Exception:
                    for future in pending:
                        future.cancel()
                    raise

                for item in parsed_items:
                    out_f.write(json.dumps({"en": item.en, "ru": item.ru, "vi": item.vi}, ensure_ascii=False) + "\n")
                    total_items += 1

                out_f.flush()
                pbar.update(1)
                pbar.set_postfix({"total_items": total_items})
                batch = next(batch_iter, None)
                if batch is not None:
                    pending.append(executor.submit(translate_batch, *batch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh tập dữ liệu song ngữ Vi-Ru bằng OpenAI-compatible API")
    parser.add_argument("--input", default="dataset/en.tatoeba.txt", help="File văn bản tiếng Anh")
    parser.add_argument("--output", default="en_ru_vi.jsonl", help="File output jsonl")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "ag/gemini-3.8-flash-low"), help="Tên model")
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", "http://192.168.2.2:20128/v1"), help="Base URL API")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""), help="API Key")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=16, help="Số luồng gọi API đồng thời (mặc định: 8)")
    parser.add_argument("--overwrite", action="store_true", help="Xóa dữ liệu cũ và chạy lại từ đầu thay vì resume")
    args = parser.parse_args()

    generate_vi_ru_from_en(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        workers=args.workers,
    )
