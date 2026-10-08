"""CPU regressions: run with python -m unittest discover -s tests -v."""
import contextlib
import io
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from ruvi_bitnet.model import BitLinear, RuViBitNet10M, RuViConfig
from ruvi_bitnet.distill import TranslationPrefixDataset, collate_prefix_fn
from scripts.export_quantized import export_model
from scripts.export_bf16 import export_bf16
from scripts.infer_ternary_logic import BitLinearLogic, load_model_ternary_logic, ternary_gemm_logic
from ruvi_bitnet.checkpoint import load_checkpoint_weights
from scripts import train as training


def small_config(**kwargs):
    values = dict(vocab_size=32, d_model=8, n_layers=3, n_heads=2, n_kv_heads=1,
                  head_dim=4, d_ff=16, max_seq_len=32, dropout=0, label_smoothing=0)
    values.update(kwargs)
    return RuViConfig(**values)


class Regressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def test_export_logic_and_scale_state_roundtrip(self):
        cfg = small_config()
        original = RuViBitNet10M(cfg).eval()
        with torch.no_grad():
            for name, parameter in original.named_parameters():
                if 'norm.weight' in name:
                    parameter.uniform_(0.5, 1.5)
        ids = torch.tensor([[4, 8, 3, 9, 2]])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            torch.save({'config': cfg, 'model_state_dict': original.state_dict()}, root / 'raw.pt')
            for packed in (False, True):
                output = root / f'{packed}.pt'
                export_model(str(root / 'raw.pt'), str(output), packed)
                checkpoint = torch.load(output, weights_only=False)
                standard = RuViBitNet10M(cfg).eval()
                load_checkpoint_weights(standard, checkpoint)
                logic = load_model_ternary_logic(str(output))
                with torch.no_grad():
                    expected = standard(ids, prefix_len=3)['logits']
                    torch.testing.assert_close(logic(ids, prefix_len=3)['logits'], expected, atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(expected, original(ids, prefix_len=3)['logits'], atol=0.005, rtol=0.02)
                for name, value in standard.state_dict().items():
                    if 'norm.weight' in name:
                        torch.testing.assert_close(logic.state_dict()[name], value, atol=0, rtol=0)
                fresh = RuViBitNet10M(cfg).eval()
                fresh.load_state_dict(standard.state_dict())
                torch.testing.assert_close(fresh(ids, prefix_len=3)['logits'], expected)
                fresh.to(dtype=torch.float64)
                for module in fresh.modules():
                    if isinstance(module, BitLinear):
                        self.assertEqual(module.export_scale.dtype, torch.float64)
                load_checkpoint_weights(standard, {'model_state_dict': original.state_dict()})
                self.assertTrue(all(m.export_scale is None for m in standard.modules() if isinstance(m, BitLinear)))
                torch.save({'config': cfg, 'model_state_dict': fresh.state_dict()}, root / 'resaved.pt')
                with self.assertRaises(ValueError):
                    export_model(str(root / 'resaved.pt'), str(root / 'invalid.pt'))
                with self.assertRaises(ValueError):
                    export_bf16(str(root / 'resaved.pt'), str(root / 'invalid.pt'))
                invalid = dict(checkpoint, model_state_dict=dict(checkpoint['model_state_dict']))
                invalid['model_state_dict'].pop('layers.1.attn_norm.weight')
                with self.assertRaises(RuntimeError):
                    load_checkpoint_weights(RuViBitNet10M(cfg), invalid)

    def test_chunked_integer_kernel_and_bias(self):
        x = torch.randint(-128, 128, (2, 19, 259), dtype=torch.int32)
        weight = torch.randint(-1, 2, (35, 259), dtype=torch.int8)
        expected = x.long() @ weight.long().T
        torch.testing.assert_close(ternary_gemm_logic(x, weight), expected, atol=0, rtol=0)
        layer = BitLinear(259, 35, bias=True)
        layer.export_scale = torch.tensor(0.025)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.uniform_(-0.2, 0.2)
        logic = BitLinearLogic(259, 35, bias=True)
        logic.load_ternary(weight, layer.export_scale)
        with torch.no_grad():
            logic.bias.copy_(layer.bias)
        inputs = torch.randn(2, 19, 259)
        torch.testing.assert_close(logic(inputs), layer(inputs), atol=3e-6, rtol=2e-5)
        self.assertEqual(logic.weight.element_size(), 1)
        with self.assertRaises(ValueError):
            logic.load_ternary(torch.full_like(weight, 2), torch.tensor(1.0))

    def test_dataset_truncates_to_max_seq_len(self):
        cfg = small_config(max_seq_len=12)
        samples = [{"src_ids": list(range(6, 32)) * 2, "tgt_ids": list(range(6, 32)), "direction": "vi2ru"}]
        item = TranslationPrefixDataset(samples, cfg)[0]
        self.assertEqual(len(item["input_ids"]), cfg.max_seq_len)
        self.assertEqual(item["input_ids"][0].item(), cfg.tag_ru_id)
        self.assertEqual(item["input_ids"][-1].item(), cfg.eos_token_id)
        self.assertEqual(item["input_ids"][item["prefix_len"] - 1].item(), cfg.sep_token_id)
        batch = collate_prefix_fn([item], pad_token_id=cfg.pad_token_id)
        out = RuViBitNet10M(cfg).eval()(input_ids=batch["input_ids"], mask=batch["mask"], labels=batch["labels"])
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_kd_ignores_prefix_padding_and_batch_size(self):
        model = RuViBitNet10M(small_config()).eval()
        ids = torch.tensor([[4, 8, 3, 9, 2]])
        labels = torch.tensor([[-100, -100, -100, 9, 2]])
        teacher = torch.randn(1, 5, 32)
        expected = model(ids, prefix_len=3, labels=labels, teacher_logits=teacher)['loss']
        changed = teacher.clone()
        changed[:, :2] = float('nan')
        actual = model(ids, prefix_len=3, labels=labels, teacher_logits=changed)['loss']
        torch.testing.assert_close(actual, expected)
        actual = model(ids.repeat(2, 1), prefix_len=3, labels=labels.repeat(2, 1),
                       teacher_logits=teacher.repeat(2, 1, 1))['loss']
        torch.testing.assert_close(actual, expected)
        padded_ids = torch.cat([ids, torch.zeros(1, 3, dtype=torch.long)], 1)
        padded_labels = torch.cat([labels, torch.full((1, 3), -100)], 1)
        padded_teacher = torch.cat([teacher, torch.full((1, 3, 32), float('nan'))], 1)
        actual = model(padded_ids, prefix_len=3, labels=padded_labels, teacher_logits=padded_teacher)['loss']
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            model(ids, labels=torch.full_like(ids, -100))

    def test_generation_zero_context_and_independent_eos(self):
        cfg = small_config(n_layers=0, max_seq_len=8)
        model = RuViBitNet10M(cfg).eval()
        prompt = torch.tensor([[4, 3], [5, 3]])
        torch.testing.assert_close(model.generate(prompt, 0), prompt)
        def scores(tokens):
            result = torch.full((2, 1, cfg.vocab_size), -100.0)
            for i, token in enumerate(tokens):
                result[i, 0, token] = 100
            return result
        with patch('ruvi_bitnet.model.F.linear', side_effect=[scores([2, 7]), scores([8, 2])]):
            output = model.generate(prompt, 4)
        self.assertEqual(output[:, 2:].tolist(), [[2, 0], [7, 2]])
        with self.assertRaises(ValueError):
            model.generate(prompt, 7)
        with self.assertRaises(ValueError):
            model(torch.ones(1, 9, dtype=torch.long))
        with self.assertRaises(ValueError):
            model.rope.apply_rope(torch.zeros(1, 2, 1, cfg.head_dim), 2, offset=7)

    def test_cached_decode_matches_full_forward(self):
        model = RuViBitNet10M(small_config()).eval()
        prompt = torch.tensor([[4, 8, 3]])
        generated = model.generate(prompt, 4, eos_token_id=-1)
        predictions = model(generated, prefix_len=3)['logits'].argmax(-1)
        torch.testing.assert_close(generated[:, 3:], predictions[:, 2:6], atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_scale_move_logic_and_bf16_backward(self):
        cfg = small_config()
        standard = RuViBitNet10M(cfg).eval()
        for module in standard.modules():
            if isinstance(module, BitLinear):
                with torch.no_grad():
                    beta = module.weight.abs().mean().clamp(min=1e-5)
                    module.weight.copy_((module.weight / beta).round().clamp(-1, 1))
                    module.export_scale = beta
        standard.to('cuda')
        for module in standard.modules():
            if isinstance(module, BitLinear):
                self.assertEqual(module.export_scale.device.type, 'cuda')
        ids = torch.tensor([[4, 8, 3, 9, 2]], device='cuda')
        logic = RuViBitNet10M(cfg).to('cuda').eval()
        logic.load_state_dict(standard.state_dict())
        for name, module in list(logic.named_modules()):
            if isinstance(module, BitLinear):
                parent, child = name.rsplit('.', 1)
                replacement = BitLinearLogic(module.in_features, module.out_features).to('cuda')
                replacement.load_ternary(module.weight, module.export_scale)
                setattr(logic.get_submodule(parent), child, replacement)
        with torch.no_grad():
            torch.testing.assert_close(logic(ids, prefix_len=3)['logits'],
                                       standard(ids, prefix_len=3)['logits'], atol=2e-6, rtol=2e-5)
        if torch.cuda.is_bf16_supported():
            baseline = RuViBitNet10M(small_config(quantize_hidden_layers=False)).to('cuda').train()
            labels = torch.tensor([[-100, -100, -100, 9, 2]], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = baseline(ids, prefix_len=3, labels=labels)['loss']
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in baseline.parameters() if p.grad is not None))

    def test_adjusted_resume_lr(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0, 1 - s / 4))
        optimizer.param_groups[0]['lr'] = 0
        training.restore_scheduler(scheduler, optimizer, {'step': 2, 'total_steps': 2}, 4, 0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.005)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.005)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.0025)

    def test_real_train_resume_and_token_weighted_validation(self):
        samples = [{'src_ids': [8] * (i % 3 + 1), 'tgt_ids': [9] * (i % 4 + 1),
                    'direction': 'vi2ru'} for i in range(10)]
        class Tokenizer:
            pad_token_id, bos_token_id, eos_token_id, unk_token_id = 0, 1, 2, 6
            def __len__(self): return 32
            def convert_tokens_to_ids(self, token): return {'<sep>': 3, '<2ru>': 4, '<2vi>': 5}[token]
        def config_factory(**kwargs):
            return small_config(**kwargs, d_model=8, d_ff=16, n_layers=3, n_heads=2, n_kv_heads=1, head_dim=4)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(training.PreTrainedTokenizerFast, 'from_pretrained', return_value=Tokenizer()), \
                patch.object(training, 'load_dataset', side_effect=lambda *_: list(samples)), \
                patch.object(training, 'RuViConfig', side_effect=config_factory), \
                patch.object(training.torch.cuda, 'is_available', return_value=False):
            root = Path(directory)
            kwargs = dict(tokenizer_dir=directory, epochs=2, batch_size=3, val_split=0.3,
                          dropout=0.2, label_smoothing=0.1, report_stats=False, max_seq_len=32)
            training.train(output_model=str(root / 'full.pt'), **kwargs)
            original_save = training.save_checkpoint
            def stop_after_epoch_one(path, state):
                original_save(path, state)
                if path.endswith('_last.pt') and state['epoch'] == 1:
                    raise InterruptedError('simulate process restart')
            with patch.object(training, 'save_checkpoint', side_effect=stop_after_epoch_one):
                with self.assertRaises(InterruptedError):
                    training.train(output_model=str(root / 'resumed.pt'), **kwargs)
            training.train(output_model=str(root / 'resumed.pt'), resume=str(root / 'resumed_last.pt'), **kwargs)
            full = torch.load(root / 'full_last.pt', weights_only=False)
            resumed = torch.load(root / 'resumed_last.pt', weights_only=False)
            for key, value in full['model_state_dict'].items():
                torch.testing.assert_close(resumed['model_state_dict'][key], value, atol=0, rtol=0)
            self.assertEqual(resumed['scheduler_state_dict'], full['scheduler_state_dict'])
            torch.testing.assert_close(resumed['torch_rng'], full['torch_rng'], atol=0, rtol=0)
            self.assertEqual(resumed['validation_nll'], full['validation_nll'])
            model = RuViBitNet10M(full['config']).eval()
            model.load_state_dict(full['model_state_dict'])
            # Rebatch validation and compare to one combined token-weighted batch.
            validation_samples = list(samples)
            random.Random(42).shuffle(validation_samples)
            loader = DataLoader(TranslationPrefixDataset(validation_samples[:3], full['config']), batch_size=2,
                                collate_fn=collate_prefix_fn)
            weighted, tokens = 0.0, 0
            with torch.no_grad():
                for batch in loader:
                    result = model(batch['input_ids'], mask=batch['mask'], labels=batch['labels'])
                    n = result['target_tokens'].item()
                    weighted += result['loss'].item() * n
                    tokens += n
            self.assertAlmostEqual(weighted / tokens, full['validation_nll'], places=6)


if __name__ == '__main__':
    unittest.main()
