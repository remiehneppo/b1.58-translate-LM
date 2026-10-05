"""Run: python -m unittest discover -s tests -p test_teacher_concurrency.py"""
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import generate_teacher_openai as teacher


class TeacherConcurrencyTest(unittest.TestCase):
    def test_input_is_read_lazily(self):
        class BoundedInput(io.StringIO):
            count = 0

            def __next__(self):
                line = super().__next__()
                self.count += 1
                if self.count > 6:
                    raise AssertionError('Input was read beyond the worker window')
                return line

        source = BoundedInput('\n'.join(map(str, range(100))))
        real_open = open

        def tracked_open(path, *args, **kwargs):
            return source if path == 'streaming-input' else real_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(teacher, 'open', side_effect=tracked_open), patch.object(teacher, 'OpenAI') as factory, patch.object(teacher, 'sleep'):
            factory.return_value.chat.completions.create.side_effect = RuntimeError('LLM unavailable')
            with self.assertRaisesRegex(RuntimeError, 'LLM unavailable'):
                teacher.generate_vi_ru_from_en('streaming-input', str(Path(directory) / 'out.jsonl'), api_key='test-key', workers=2, batch_size=3)
        self.assertEqual(source.count, 6)
        self.assertTrue(source.closed)

    def test_retry_delays_and_limit(self):
        valid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({'items': [{'en': 'hello', 'ru': 'привет', 'vi': 'xin chào'}]})
        ))])
        invalid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='invalid'))])
        with tempfile.TemporaryDirectory() as directory, patch.object(teacher, 'OpenAI') as factory, patch.object(teacher, 'sleep') as sleep:
            source = Path(directory) / 'en.txt'
            output = Path(directory) / 'triplets.jsonl'
            source.write_text('hello')
            create = factory.return_value.chat.completions.create
            create.side_effect = [RuntimeError('LLM unavailable'), invalid, valid]
            teacher.generate_vi_ru_from_en(str(source), str(output), api_key='test-key', workers=1)
            self.assertEqual(create.call_count, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10])
            self.assertEqual(len(output.read_text().splitlines()), 1)
            self.assertEqual(factory.call_args.kwargs['max_retries'], 0)
            create.reset_mock()
            sleep.reset_mock()
            create.side_effect = RuntimeError('LLM unavailable')
            with self.assertRaisesRegex(RuntimeError, 'LLM unavailable'):
                teacher.generate_vi_ru_from_en(str(source), str(output), api_key='test-key', workers=1, overwrite=True)
            self.assertEqual(create.call_count, 11)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], list(range(5, 51, 5)))
            self.assertEqual(output.read_text(), '')

    def test_concurrency_order_failure_and_resume(self):
        lock = threading.Lock()
        active = peak = 0
        fail = None
        seen = []

        def create(**kwargs):
            nonlocal active, peak
            chunk = json.loads(kwargs['messages'][1]['content'].split('English sentences:\n')[1])
            with lock:
                active += 1
                peak = max(peak, active)
                seen.extend(chunk)
            try:
                time.sleep(0.03 if chunk[0] == '0' else 0.005)
                if fail in chunk:
                    raise RuntimeError('simulated API failure')
                content = json.dumps({'items': [{'en': x, 'vi': 'vi' + x, 'ru': 'ru' + x} for x in chunk]})
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            finally:
                with lock:
                    active -= 1

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as directory, patch.object(teacher, 'OpenAI', return_value=client), patch.object(teacher, 'sleep'):
            source = Path(directory) / 'en.txt'
            output = Path(directory) / 'pairs.jsonl'
            source.write_text('\n'.join(map(str, range(20))))
            teacher.generate_vi_ru_from_en(str(source), str(output), api_key="test-key", batch_size=1)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(1 < peak <= 8)
            self.assertEqual(rows, [{'en': str(i), 'ru': 'ru' + str(i), 'vi': 'vi' + str(i)} for i in range(20)])
            self.assertEqual(len(rows), 20)
            fail = '2'
            with self.assertRaises(RuntimeError):
                teacher.generate_vi_ru_from_en(str(source), str(output), api_key="test-key", batch_size=1, workers=3, overwrite=True)
            self.assertEqual(len(output.read_text().splitlines()), 2)
            fail = None
            seen.clear()
            teacher.generate_vi_ru_from_en(str(source), str(output), api_key="test-key", batch_size=1, workers=3)
            self.assertEqual(sorted(map(int, seen)), list(range(2, 20)))
            self.assertEqual(len(output.read_text().splitlines()), 20)
            output.write_text(json.dumps({'src': 'vi0', 'tgt': 'ru0', 'dir': 'vi2ru'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'en/ru/vi'):
                teacher.generate_vi_ru_from_en(str(source), str(output), api_key="test-key")
            for kwargs in ({'workers': 0}, {'batch_size': 0}):
                with self.assertRaises(ValueError):
                    teacher.generate_vi_ru_from_en(str(source), str(output), api_key="test-key", **kwargs)


if __name__ == '__main__':
    unittest.main()
