"""Compare native greedy token IDs and every decode logit against a CPU PyTorch oracle."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from scripts.export_native import load_native_reference


def check_native(checkpoint, image, executable, prompt, max_new=8, context=128,
                 float_storage='bf16', atol=0.02, rtol=0.005):
    model = load_native_reference(checkpoint, float_storage)
    cfg = model.config
    ids = torch.tensor([prompt], dtype=torch.long)
    expected_logits = []
    # The final RMSNorm executes once per prediction in generate(), including prefill.
    def capture(_module, _inputs, output):
        expected_logits.append(torch.nn.functional.linear(output[:, -1:], model.embed_tokens.weight)[0, 0].detach())
    handle = model.norm.register_forward_hook(capture)
    try:
        expected_ids = model.generate(ids, max_new_tokens=max_new)[0, len(prompt):].tolist()
    finally:
        handle.remove()
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / 'logits.bin'
        result = subprocess.run([str(executable), '--model', str(image), '--prompt', ','.join(map(str, prompt)),
                                 '--context', str(context), '--max-new', str(max_new), '--logits', str(trace)],
                                check=True, capture_output=True, text=True)
        native = json.loads(result.stdout)
        raw = trace.read_bytes()
    if native['generated_ids'] != expected_ids:
        raise AssertionError(f"Greedy mismatch: native={native['generated_ids']}, reference={expected_ids}")
    expected = torch.stack(expected_logits) if expected_logits else torch.empty(0, cfg.vocab_size)
    actual = torch.tensor(struct.unpack('<' + 'f' * (len(raw) // 4), raw)).reshape(expected.shape)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return dict(status='pass', reference='CPU FP32 compute on native storage weights',
                float_storage=float_storage, max_abs_logit_error=(actual - expected).abs().max().item() if actual.numel() else 0,
                atol=atol, rtol=rtol, native=native)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--executable', default='native/build/ruvi-native')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prompt', help='Comma-separated prefix token IDs')
    inputs.add_argument('--text')
    parser.add_argument('--tokenizer', default='ru-vi-tatoeba-demo')
    parser.add_argument('--direction', choices=['vi2ru', 'ru2vi'], default='vi2ru')
    parser.add_argument('--float-storage', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--max-new', type=int, default=8)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--atol', type=float, default=0.02)
    parser.add_argument('--rtol', type=float, default=0.005)
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(2)
    tokenizer = None
    if args.text is not None:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
        cfg = load_native_reference(args.checkpoint, args.float_storage).config
        if len(tokenizer) != cfg.vocab_size:
            parser.error('tokenizer vocabulary size does not match checkpoint')
        prompt = [cfg.tag_ru_id if args.direction == 'vi2ru' else cfg.tag_vi_id,
                  *tokenizer.encode(args.text, add_special_tokens=False), cfg.sep_token_id]
    else:
        prompt = [int(token) for token in args.prompt.split(',')]
    report = check_native(args.checkpoint, args.image, args.executable, prompt, args.max_new,
                          args.context, args.float_storage, args.atol, args.rtol)
    if tokenizer is not None:
        report['translation'] = tokenizer.decode(report['native']['generated_ids'], skip_special_tokens=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
