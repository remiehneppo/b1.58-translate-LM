"""Native parity and malformed-image checks; builds a temporary host executable."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib

import torch
from ruvi_bitnet.model import RuViBitNet10M, RuViConfig
from scripts.export_native import export_native
from scripts.check_native import check_native
from scripts.benchmark_native import benchmark

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('g++'), 'g++ unavailable')
class NativeRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.executable = cls.root / 'ruvi-native'
        subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off', str(ROOT / 'native/main.cpp'),
                        str(ROOT / 'native/ruvi.cpp'), '-o', str(cls.executable)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def fixture(self, **values):
        options = dict(vocab_size=19, d_model=7, n_layers=3, n_heads=2,
                       n_kv_heads=1, head_dim=4, d_ff=13, max_seq_len=32, dropout=0)
        options.update(values)
        torch.manual_seed(42)
        model = RuViBitNet10M(RuViConfig(**options)).eval()
        with torch.no_grad():
            for name, value in model.named_parameters():
                if 'norm.weight' in name:
                    value.uniform_(0.75, 1.25)
        checkpoint = self.root / 'fixture.pt'
        torch.save({'config': model.config, 'model_state_dict': model.state_dict()}, checkpoint)
        return checkpoint

    def test_parity_precision_layer_flags_and_bidirectional_prefix(self):
        variants = [{}, {'quantize_first_layer': True, 'quantize_last_layer': True},
                    {'quantize_hidden_layers': False}, {'n_kv_heads': 2}, {'n_layers': 0}]
        for variant in variants:
            for storage in ('fp32', 'bf16'):
                with self.subTest(variant=variant, storage=storage):
                    checkpoint = self.fixture(**variant)
                    image = self.root / 'fixture.bin'
                    export_native(checkpoint, image, storage)
                    for prompt in ([4, 8, 3], [5, 7, 9, 8, 10, 3]):
                        report = check_native(checkpoint, image, self.executable, prompt, 5, 16,
                                              storage, atol=2e-5, rtol=0.002)
                        self.assertEqual(report['status'], 'pass')

    def test_benchmark_same_image_and_workspace_accounting(self):
        checkpoint = self.fixture()
        image = self.root / 'fixture.bin'
        manifest = export_native(checkpoint, image)
        report = benchmark(self.executable, image, (16,), (3,), max_new=4, repeats=2)
        self.assertEqual(report['sha256'], manifest['sha256'])
        run = report['runs'][0]
        self.assertEqual(run['image_crc32'], manifest['crc32'])
        self.assertEqual(run['repeats'], 2)
        self.assertEqual(run['prompt_tokens'], 3)
        self.assertEqual(run['model_bytes'], image.stat().st_size)
        self.assertGreater(run['workspace_bytes'], 0)
        with self.assertRaises(ValueError):
            benchmark(self.executable, image, (16,), (16,), max_new=1)

    def test_eos_stops_without_decode(self):
        checkpoint = self.fixture(eos_token_id=0)
        state = torch.load(checkpoint, weights_only=False)
        state['model_state_dict']['embed_tokens.weight'].zero_()
        torch.save(state, checkpoint)
        image = self.root / 'fixture.bin'
        export_native(checkpoint, image)
        report = check_native(checkpoint, image, self.executable, [4, 8, 3], 5, 16, atol=2e-5)
        self.assertEqual(report['native']['generated_ids'], [0])
        self.assertEqual(report['native']['decode_steps'], 0)
        self.assertIsNone(report['native']['decode_tokens_per_second'])

    def test_image_validation_context_invalid_tokens_and_zero_generation(self):
        checkpoint = self.fixture()
        image = self.root / 'fixture.bin'
        export_native(checkpoint, image)
        original = image.read_bytes()
        def run(payload, *extra):
            image.write_bytes(payload)
            return subprocess.run([str(self.executable), '--model', str(image), '--context', '16',
                                   '--prompt', '4,8,3', '--max-new', '5', *extra], capture_output=True)
        self.assertNotEqual(run(original[:79]).returncode, 0)
        self.assertNotEqual(run(original[:-1]).returncode, 0)
        corrupt = bytearray(original); corrupt[-20] ^= 1
        self.assertNotEqual(run(corrupt).returncode, 0)
        corrupt = bytearray(original); corrupt[56] ^= 1  # finite theta corruption must fail CRC.
        self.assertNotEqual(run(corrupt).returncode, 0)
        corrupt = bytearray(original); struct.pack_into('<I', corrupt, 8, 65535)
        struct.pack_into('<I', corrupt, 68, 0); struct.pack_into('<I', corrupt, 68, zlib.crc32(corrupt))
        self.assertNotEqual(run(corrupt).returncode, 0)  # wrong tensor dimensions despite valid CRC.
        corrupt = bytearray(original); struct.pack_into('<H', corrupt, 112, 0x7f80)  # BF16 +inf.
        struct.pack_into('<I', corrupt, 68, 0); struct.pack_into('<I', corrupt, 68, zlib.crc32(corrupt))
        self.assertNotEqual(run(corrupt).returncode, 0)
        corrupt = bytearray(original); offset = 80
        while offset < len(corrupt):
            kind = struct.unpack_from('<I', corrupt, offset + 8)[0]
            size = struct.unpack_from('<I', corrupt, offset + 16)[0]
            if kind == 2:
                corrupt[offset + 32] |= 3  # invalid ternary code, with a valid checksum.
                break
            offset += 32 + ((size + 15) // 16) * 16
        struct.pack_into('<I', corrupt, 68, 0); struct.pack_into('<I', corrupt, 68, zlib.crc32(corrupt))
        self.assertNotEqual(run(corrupt).returncode, 0)
        self.assertNotEqual(run(original, '--prompt', '4,99,3').returncode, 0)
        self.assertNotEqual(run(original, '--max-new', '14').returncode, 0)
        self.assertNotEqual(run(original, '--context', '33').returncode, 0)
        result = run(original, '--max-new', '0')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)['generated_ids'], [])
        result = run(original, '--repeat', '2')
        self.assertEqual(result.returncode, 0)
        self.assertGreater(json.loads(result.stdout)['workspace_bytes'], 0)
        # partition padding is allowed, but never interpreted as model weights.
        self.assertEqual(run(original + bytes(64)).returncode, 0)


if __name__ == '__main__':
    unittest.main()
