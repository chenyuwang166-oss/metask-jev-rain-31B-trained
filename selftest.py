"""Synthetic offline selftest. Never imports a model runtime or opens a socket."""
import argparse
import copy
import hashlib
import json
import math
import re
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import backends
import core
import launch
import pkg_check


class WordTokenizer:
    """Reversible synthetic tokenizer: one token per whitespace-prefixed word."""
    def __init__(self):
        self.ids, self.pieces = {}, []

    def _id(self, piece):
        if piece not in self.ids:
            self.ids[piece] = len(self.pieces)
            self.pieces.append(piece)
        return self.ids[piece]

    def encode(self, text, add_special_tokens=False):
        return [self._id(p) for p in re.findall(r'\s*\S+|\s+', text)]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(self.pieces[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        return '<turn>user ' + messages[0]['content'] + ' <turn>model'


def synthetic_task(words):
    """Invented rules + narrative of about `words` tokens; a yes/no question about the rules."""
    rules = 'Rules:\nR1: A crate is late if it reaches dock B after day 30.\nR2: Late crates are refunded.\n\n'
    sentences, day = [], 0
    while sum(len(s.split()) for s in sentences) < words:
        day += 1
        sentences.append('Day %d: the courier moved crate C%d from dock A to dock B without delay.' % (day, day))
    state = rules + 'Events:\n' + ' '.join(sentences)
    payload = {'state': state, 'questions': {'q': {'type': 'noul', 'instructions': 'Is crate C7 refunded?'}}}
    return core.normalize_request(payload, 1)[0][1]


class SelfTest(unittest.TestCase):
    def test_package_integrity(self):
        self.assertIn(pkg_check.checks()['status'], ('PREPARATION_CHECK_PASS', 'FROZEN_PACKAGE_CHECK_PASS'))

    def test_pending_cannot_serve(self):
        config = pkg_check.read('config.json')
        if not config['package'].get('nonformal_fitted'):
            with self.assertRaisesRegex(ValueError, 'fit pending'):
                launch.check_preparation(allow_nonformal=True)
        if config['package'].get('finalized'):
            launch.check_preparation()  # a frozen package starts formally, without any extra switch
        else:
            with self.assertRaises(ValueError):
                launch.check_preparation()

    def test_offline_environment(self):
        state = Path(Path.cwd().anchor) / 'jev-31b-offline-state'
        env = launch.child_environment({}, state, state / 'startup')
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'VLLM_BATCH_INVARIANT'):
            self.assertEqual(env[key], '1')
        self.assertEqual(env['VLLM_USE_FLASHINFER_SAMPLER'], '0')
        self.assertTrue(Path(env['TMPDIR']).is_relative_to(state))

    def test_served_mode_is_the_stated_policy(self):
        policy = pkg_check.SERVICE_POLICY
        self.assertEqual(launch.SERVED_MODE, policy['mode'])
        c = pkg_check.read('config.json')
        self.assertEqual((c['routing']['mode'], c['routing']['quota'], c['routing']['target_slow_share']),
                         (policy['mode'], policy['quota'], policy['quota']))
        self.assertEqual(pkg_check.policy_problems(policy), [])

    def test_yes_no_constant(self):
        p = pkg_check.SERVICE_POLICY['noul_p_cal']
        self.assertTrue(0.81 <= p <= 0.95)
        cal = pkg_check.read('calibration.json')['models'][pkg_check.MODEL_KEY]
        for side in ('fast', 'slow', 'fallback'):
            block = core.validate_calibration({'models': {'m': cal}}, 'm', side)
            self.assertEqual(block['noul_calibration']['p_cal'], p)
            out = core.probabilities(['no', 'yes'], {'no': 0.2, 'yes': 1.5}, 'noul', block)
            self.assertEqual(out, {'no': 1 - p, 'yes': p})
            out = core.probabilities(['no', 'yes'], {'no': 1.5, 'yes': 0.2}, 'noul', block)
            self.assertEqual(out, {'no': p, 'yes': 1 - p})
        for p_bad in (0.80, 0.951):
            policy = dict(pkg_check.SERVICE_POLICY, noul_p_cal=p_bad)
            self.assertIn('yes/no constant outside [0.81, 0.95]', pkg_check.policy_problems(policy))

    def test_drafting_budget_inside_the_context(self):
        policy = pkg_check.SERVICE_POLICY
        c = pkg_check.read('config.json')
        served = copy.deepcopy(c)
        core.apply_profile(served)
        core.apply_budget_tier(served, None)
        identity = core.threshold_identity(served)
        self.assertEqual((identity['max_new_tokens'], identity['reasoning_words']),
                         (policy['draft']['max_new_tokens'], policy['draft']['reasoning_words']))
        self.assertEqual(served['generation']['max_new_tokens'], policy['draft']['max_new_tokens'])
        self.assertEqual(served['routing']['slow_max_prompt_tokens'], policy['slow_max_prompt_tokens'])
        if policy['mode'] == 'routed':
            self.assertLessEqual(policy['slow_max_prompt_tokens'] + policy['draft']['max_new_tokens']
                                 + pkg_check.SLOW_READOUT_MARGIN, c['model']['vllm_options']['max_model_len'])
            wide = dict(policy, slow_max_prompt_tokens=6653)
            self.assertIn('drafting budget outside the vLLM context', pkg_check.policy_problems(wide))

    def test_slow_type_exclusion(self):
        """Excluded types (v3 core patch) never draft; the other types do (uniform mode, synthetic readout)."""
        policy = pkg_check.SERVICE_POLICY
        c = pkg_check.read('config.json')
        excluded = policy['slow_excluded_types']
        self.assertEqual(c['routing'].get('slow_excluded_types', []), excluded)
        if not excluded:
            return
        self.assertEqual(getattr(core, 'SLOW_TYPE_EXCLUSION', None), 'routing.slow_excluded_types')

        class DraftingBackend:
            def slow_session(self, ids, unparsed_policy, meter, max_new_tokens=None):
                return None, False  # a draft without answer marker: the first-pass answer is kept

        def fast(backend, task, config, ids, labels, groups, meter):
            return {label: -float(index) for index, label in enumerate(labels)}, 0, None
        length = {key: False for key in ('truncated', 'character_precompressed', 'protected_content_exceeds_budget',
                                         'physical_truncated', 'long_input', 'slow_prompt_exceeds_budget')}
        length['original_prompt_tokens'] = 1
        cal = pkg_check.read('calibration.json')
        with tempfile.TemporaryDirectory() as tmp:
            config = copy.deepcopy(c)
            core.apply_profile(config)
            config['routing'].update(state_dir=str(Path(tmp).resolve()), run_id='exclusion', resume=False,
                                     mode='uniform', quota=1., fuse=1.)
            import quota
            service = core.Service(config, cal, DraftingBackend(), quota.RuntimeCounter(config))
            routes = {}
            with patch('core.prepare_task', return_value=(None, (None, {}, [1], [1]), length)), \
                    patch('core.fast_attempt_with_retry', side_effect=fast):
                for kind, question in (('noul', {'type': 'noul', 'instructions': 'Is it late?'}),
                                       ('choice', {'type': 'choice', 'instructions': 'Which one?',
                                                   'criteria': {'red': 'r', 'blue': 'b'}})):
                    response = service.answer({'state': 'synthetic', 'questions': {'q': question}})
                    routes[kind] = response['usage']['questions']['q']
            for kind in ('choice', 'noul', 'score'):
                if kind not in routes:
                    continue
                with self.subTest(kind=kind):
                    if kind in excluded:
                        self.assertEqual((routes[kind]['path'], routes[kind]['route_reason']), ('fast', 'type_excluded'))
                    else:
                        self.assertEqual(routes[kind]['path'], 'slow')
        bad = copy.deepcopy(c)
        bad['routing']['slow_excluded_types'] = ['noul', 'noul']
        with self.assertRaisesRegex(ValueError, 'slow_excluded_types'):
            core.validate_config(bad)

    def test_threshold_identity(self):
        c = pkg_check.read('config.json')
        expected = core.threshold_identity(c)
        self.assertEqual(expected['model_key'], pkg_check.MODEL_KEY)
        self.assertEqual(expected['max_prompt_tokens'], pkg_check.INPUT_BUDGET)
        c['routing']['margin_threshold_fitted_for'] = {'placeholder': False, **expected}
        self.assertEqual(core.validate_threshold(c), [])
        c['routing']['margin_threshold_fitted_for']['model_key'] = 'wrong'
        if c['routing']['mode'] == 'routed':
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                core.validate_threshold(c)  # routed serving refuses a threshold fitted for another identity
        else:
            self.assertTrue(core.validate_threshold(c))

    def test_adapter_reference_materializes_into_served_identity(self):
        c = pkg_check.read('config.json')
        identity = core.threshold_identity(c)
        c['routing']['margin_threshold_fitted_for'] = {'placeholder': False, **identity}
        c['profiles'][pkg_check.MODEL_KEY]['routing'] = copy.deepcopy(c['routing'])
        folder = Path(tempfile.gettempdir()) / 'synthetic-model' / 'adapter'
        self.assertEqual(c['model']['adapter'], launch.ADAPTER_REF)
        served = launch.materialize_adapter(c, folder)
        self.assertEqual(served['model']['adapter'], str(folder))
        self.assertEqual(served['routing']['margin_threshold_fitted_for']['adapter'], str(folder))
        self.assertEqual(core.validate_threshold(served), [])
        self.assertEqual(c['model']['adapter'], launch.ADAPTER_REF)
        with self.assertRaises(ValueError):
            launch.materialize_adapter(c, None)
        raw = copy.deepcopy(c)
        raw['model']['adapter'] = str(folder)
        with self.assertRaises(ValueError):
            launch.materialize_adapter(raw, folder)

    def test_adapter_folder_is_exactly_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            folder = model / 'adapter'
            folder.mkdir()
            payload = {'adapter_config.json': b'{"r": 16}\n', 'adapter_model.safetensors': b'synthetic weights'}
            pinned = {}
            for name, data in payload.items():
                (folder / name).write_bytes(data)
                pinned[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
            with patch.object(launch, 'ADAPTER_FILES', pinned):
                self.assertEqual(launch.check_adapter(model), folder.resolve())
                (folder / 'extra.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'exactly'):
                    launch.check_adapter(model)
                (folder / 'extra.json').unlink()
                (folder / 'adapter_model.safetensors').write_bytes(b'tampered weights!')
                with self.assertRaisesRegex(ValueError, 'differs'):
                    launch.check_adapter(model)
            with patch.object(launch, 'ADAPTER_FILES', {}):
                with self.assertRaisesRegex(ValueError, 'pending'):
                    launch.check_adapter(model)

    def test_input_budget_6653_everywhere(self):
        c = pkg_check.read('config.json')
        served = copy.deepcopy(c)
        core.apply_profile(served)
        core.apply_budget_tier(served, None)
        self.assertEqual(served['limits']['max_prompt_tokens'], 6653)
        self.assertEqual(served['limits']['physical_max_prompt_tokens'], 6653)
        for tier in ('normal', 'long'):
            self.assertEqual(served['routing']['tiers'][tier]['max_prompt_tokens'], 6653)
        self.assertLessEqual(6653 + 3, c['model']['vllm_options']['max_model_len'])

    def test_long_inputs_are_compressed_not_refused(self):
        """80k- and 7k-token synthetic inputs are answered from at most 6653 prompt tokens."""
        c = pkg_check.read('config.json')
        core.apply_profile(c)
        backend = backends.TokenizerMixin()
        backend.tokenizer = WordTokenizer()
        for words, precompressed in ((80000, True), (7000, False)):
            task = synthetic_task(words)
            _, prepared, length = core.prepare_task(task, backend, c)
            self.assertTrue(length['long_input'])
            self.assertEqual(length['character_precompressed'], precompressed)
            if not precompressed:
                self.assertGreater(length['original_prompt_tokens'], 6653)
            self.assertLessEqual(length['kept_prompt_tokens'], 6653)
            self.assertLessEqual(len(prepared[2]), 6653)
            self.assertFalse(length['physical_truncated'])
            self.assertFalse(length['protected_content_exceeds_budget'])
        task = synthetic_task(3000)
        _, prepared, length = core.prepare_task(task, backend, c)
        self.assertFalse(length['long_input'])
        self.assertFalse(length['truncated'])

    def test_readout_probability_contract(self):
        block = {'temperature_by_kind': {'choice': 1., 'score': 1., 'noul': 'one_bin'}, 'noul_calibration': {'p_cal': .81}}
        for kind, labels in [('choice', ['A', 'B', 'C']), ('score', ['1', '2', '3']), ('noul', ['no', 'yes'])]:
            logits = {label: float(i) for i, label in enumerate(labels)}
            p = core.probabilities(labels, logits, kind, block)
            self.assertEqual(set(p), set(labels))
            self.assertAlmostEqual(sum(p.values()), 1.)
            self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in p.values()))
            self.assertEqual(max(labels, key=p.__getitem__), labels[-1])

    def test_readout_rejects_missing_and_nonfinite(self):
        for logits in ({0: 0.}, {0: 0., 1: float('nan')}, {0: 0., 1: float('inf')}):
            with self.assertRaises(RuntimeError):
                core.aggregate(logits, {'a': [0], 'b': [1]})

    def test_config_does_not_accept_invalid_quota(self):
        c = copy.deepcopy(pkg_check.read('config.json'))
        c['routing']['quota'] = .9
        with self.assertRaises(ValueError):
            core.validate_config(c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline', action='store_true', help='default; compatibility flag')
    p.parse_args()
    with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden in offline selftest')):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest))
    print(json.dumps({'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests': result.testsRun,
                      'synthetic': True, 'model_called': False, 'network': False}, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
