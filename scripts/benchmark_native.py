"""Run reproducible synthetic-token CPU benchmarks; does not measure translation quality."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess


def benchmark(executable, image, contexts=(64, 128, 256), lengths=(16, 64, 128), max_new=8, repeats=3):
    if len(contexts) != len(lengths) or not contexts:
        raise ValueError('contexts and prompt lengths must have equal nonzero length')
    runs = []
    for context, length in zip(contexts, lengths):
        description = json.loads(subprocess.check_output([str(executable), '--model', str(image),
                                                         '--context', str(context), '--describe'], text=True))
        if length < 2 or length + max_new > context:
            raise ValueError('prefix must include tag/sep and fit context with max-new')
        # Read tag/sep from the exporter manifest to retain the checkpoint contract.
        manifest = json.loads(Path(str(image) + '.json').read_text())
        cfg = manifest['config']
        if description['image_crc32'] != manifest['crc32'] or description['vocab'] != cfg['vocab_size']:
            raise ValueError('export manifest does not match native image')
        rng = random.Random(42)
        ids = [cfg['tag_ru_id'], *[rng.randrange(7, cfg['vocab_size']) for _ in range(length - 2)], cfg['sep_token_id']]
        result = subprocess.run([str(executable), '--model', str(image), '--context', str(context),
                                 '--prompt', ','.join(map(str, ids)), '--max-new', str(max_new),
                                 '--repeat', str(repeats)], check=True, capture_output=True, text=True)
        run = json.loads(result.stdout)
        run['prompt_ids'] = ids
        runs.append(run)
    return dict(platform=platform.platform(), machine=platform.machine(),
                workload='synthetic token IDs; no translation quality claim',
                image=str(image), sha256=hashlib.sha256(Path(image).read_bytes()).hexdigest(),
                max_new=max_new, runs=runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='native/build/ruvi-native')
    parser.add_argument('--image', required=True)
    parser.add_argument('--contexts', default='64,128,256')
    parser.add_argument('--prompt-lengths', default='16,64,128')
    parser.add_argument('--max-new', type=int, default=8)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = benchmark(args.executable, args.image, tuple(map(int, args.contexts.split(','))),
                       tuple(map(int, args.prompt_lengths.split(','))), args.max_new, args.repeat)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
