"""Benchmark best checkpoint on a test dataset and evaluate performance and output quality."""
import json
from pathlib import Path
import subprocess
import sys
import time

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "native/artifacts/ruvi_best_int8.bin"
EXECUTABLE = ROOT / "native/build/ruvi-native"
TOKENIZER_PATH = ROOT / "ru-vi-tatoeba/tokenizer.json"

TEST_SAMPLES = [
    # Vietnamese -> Russian
    {
        "src": "Xin chào, bạn khỏe không?",
        "ref": "Привет, как дела?",
        "dir": "vi2ru",
        "tag": 4,
    },
    {
        "src": "Tôi có một con bò.",
        "ref": "У меня есть корова.",
        "dir": "vi2ru",
        "tag": 4,
    },
    {
        "src": "Mở két sắt ra.",
        "ref": "Открой сейф.",
        "dir": "vi2ru",
        "tag": 4,
    },
    {
        "src": "Tôi không phải kiểu người như thế.",
        "ref": "Я не такой человек.",
        "dir": "vi2ru",
        "tag": 4,
    },
    {
        "src": "Dáng bạn đẹp đấy.",
        "ref": "У тебя красивая фигура.",
        "dir": "vi2ru",
        "tag": 4,
    },
    # Russian -> Vietnamese
    {
        "src": "Привет! Как дела?",
        "ref": "Xin chào! Bạn thế nào?",
        "dir": "ru2vi",
        "tag": 5,
    },
    {
        "src": "Зири боится быть покинутым.",
        "ref": "Ziri sợ bị bỏ rơi.",
        "dir": "ru2vi",
        "tag": 5,
    },
    {
        "src": "Я не осознавал, что происходит.",
        "ref": "Tôi không nhận ra chuyện gì đang xảy ra.",
        "dir": "ru2vi",
        "tag": 5,
    },
    {
        "src": "Антибиотики не могут убивать вирусы.",
        "ref": "Thuốc kháng sinh không thể tiêu diệt vi-rút.",
        "dir": "ru2vi",
        "tag": 5,
    },
    {
        "src": "Том не знает, куда пошла Мэри.",
        "ref": "Tom không biết Mary đã đi đâu.",
        "dir": "ru2vi",
        "tag": 5,
    },
]


def run_benchmark():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    results = []

    print("=" * 80)
    print("BỘ KIỂM TRA HIỆU NĂNG & DỊCH THUẬT (BEST CHECKPOINT - RuVi BitNet 10M)")
    print(f"Model Artifact: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print("=" * 80)

    for idx, sample in enumerate(TEST_SAMPLES):
        src_text = sample["src"]
        ref_text = sample["ref"]
        direction = sample["dir"]
        tag = sample["tag"]
        sep = 3

        encoded = tokenizer.encode(src_text)
        prompt_ids = [tag] + encoded.ids + [sep]
        prompt_str = ",".join(map(str, prompt_ids))

        # Run Tiled Kernel
        cmd = [
            str(EXECUTABLE),
            "--model", str(MODEL_PATH),
            "--prompt", prompt_str,
            "--context", "128",
            "--max-new", "16",
            "--repeat", "1",
            "--backend", "tiled",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        tiled_out = json.loads(res.stdout)

        generated_ids = tiled_out["generated_ids"]
        # Stop at EOS (token 2) if present
        if 2 in generated_ids:
            clean_ids = generated_ids[:generated_ids.index(2)]
        else:
            clean_ids = generated_ids
        translated_text = tokenizer.decode(clean_ids)

        perf = {
            "index": idx + 1,
            "direction": direction,
            "src": src_text,
            "ref": ref_text,
            "hyp": translated_text,
            "generated_ids": generated_ids,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated_ids),
            "prefill_ms": tiled_out["prefill_p50_ms"],
            "decode_ms": tiled_out["decode_p50_ms"],
            "total_ms": tiled_out["total_p50_ms"],
            "tok_per_sec": tiled_out["decode_tokens_per_second"],
        }
        results.append(perf)

        print(f"\n[Mẫu {idx+1:02d}] [{direction}]")
        print(f"  Nguồn:       {src_text}")
        print(f"  Tham chiếu:  {ref_text}")
        print(f"  Bản dịch:    {translated_text}")
        print(f"  Prefill:     {perf['prefill_ms']:.2f} ms ({perf['prompt_tokens']} tokens)")
        print(f"  Decode:      {perf['decode_ms']:.2f} ms ({perf['generated_tokens']} tokens, {perf['tok_per_sec']:.2f} tok/s)")
        print(f"  Tổng trễ:    {perf['total_ms']:.2f} ms")

    # Aggregate Statistics
    avg_tok_sec = sum(r["tok_per_sec"] for r in results if r["tok_per_sec"]) / len(results)
    avg_prefill = sum(r["prefill_ms"] for r in results) / len(results)
    avg_total = sum(r["total_ms"] for r in results) / len(results)

    print("\n" + "=" * 80)
    print("TỔNG KẾT HIỆU NĂNG TRÊN BỘ TEST:")
    print(f"  Tốc độ sinh token trung bình:  {avg_tok_sec:.2f} tokens/s")
    print(f"  Độ trễ prefill trung bình:     {avg_prefill:.2f} ms")
    print(f"  Tổng thời gian trung bình:     {avg_total:.2f} ms")
    print("=" * 80)

    report_path = ROOT / "native/artifacts/test_benchmark_results.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nĐã lưu báo cáo chi tiết vào: {report_path}")


if __name__ == "__main__":
    run_benchmark()
