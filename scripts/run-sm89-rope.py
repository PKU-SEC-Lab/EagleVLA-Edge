#!/usr/bin/env python3
"""Verify FlashRT FP16 RoPE rotation adapted to GGML NEOX layout; record action drift."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import struct
import subprocess
import tempfile
import types
import urllib.request

REFERENCE = 'results/flashrt-sm89/patch-im2col-0faleo4x/summary.json'
PATCH_RUNNER = 'scripts/run-sm89-patch-im2col.py'
PATCH_RUNNER_SHA = '95bd89b1289b8007e43f7d997cc824c60cad1edc62e55966affb07bfc38dffad'
PREFILL_RUNNER = 'scripts/run-sm89-prefill-gemm.py'
PREFILL_RUNNER_SHA = '1f655e029ab0dc513a4a90c717c03fd1d2884330cf2cca08a977403fc579d785'
GEGLU_RUNNER = 'scripts/run-sm89-geglu-port.py'
GEGLU_RUNNER_SHA = 'e963bc41c20dfc70abc493de5c98dfd2040cb57a7c1903b0d23f35176686d405'
FEATURE = 'GGML_CUDA_FLASHRT_ROPE'
COMPUTE_OVERRIDE = 'GGML_CUDA_CUBLAS_COMPUTE_TYPE'
TRACE = 'flashrt_rope:'
EXE = 'test-flashrt-rope'
MODEL_SHAPES = {(heads, rows) for heads in (1, 8) for rows in (11, 264, 268)}
PHASES = ('initial', 'positions_changed', 'input_changed')
FIXTURE_FIELDS = ('name', 'node', 'phase', 'head_dim', 'heads', 'rows',
                  'view_offset_bytes', 'eligible', 'elements', 'raw_dtype',
                  'raw_offset_bytes', 'raw_bytes', 'rope_mode', 'freq_scale',
                  'has_freq_factors', 'input_fingerprint', 'position_fingerprint',
                  'position_values')
NEW_SOURCES = ('ggml/src/ggml-cuda/flashrt-rope.cu',
               'ggml/src/ggml-cuda/flashrt-rope.cuh',
               'ggml/src/ggml-cuda/flashrt-rope-guard.h',
               'tests/test-flashrt-rope.cpp', 'scripts/run-sm89-rope.py')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_checked(path, expected):
    need(sha(path) == expected, f'helper 版本变化，未导入：{path}')
    return types.SimpleNamespace(**runpy.run_path(str(path)))


def trace_records(text):
    return [dict(re.findall(r'(\w+)=([^\s]+)', line.split(TRACE, 1)[1]))
            for line in text.splitlines() if TRACE in line]


def environment(g, saved, runtime, gpu, mode, graphs):
    # Keep accepted kernels and CPU placement; only opt into this rotation adapter.
    need(COMPUTE_OVERRIDE not in saved and COMPUTE_OVERRIDE not in os.environ,
         f'{COMPUTE_OVERRIDE} 已显式设置，与保存的基线不同；保留原设置并停止')
    env = g.environment(saved, runtime, gpu, 0, graphs, trace=True)
    env.update({FEATURE: str(mode), 'GGML_CUDA_FLASHRT_PATCH_IM2COL': '1',
                'GGML_CUDA_FLASHRT_PREFILL_GEMM': '0',
                'GGML_CUDA_FLASHRT_GEGLU': '0', 'GGML_CUDA_FLASHRT_BIAS_ACTIVATION': '0'})
    return env


def ensure_other_candidates_off(text):
    need('flashrt_prefill_gemm:' not in text and 'flashrt_geglu:' not in text and
         'flashrt_bias_activation:' not in text,
         '意外命中 prefill GEMM、GEGLU 或 bias/activation；本次只切换 RoPE')


def expected_cases():
    # head_dim, heads, rows, view bytes, eligible, rope mode, freq_scale, frequency factors.
    cases = {f'rope_h{heads}_m{rows}': (256, heads, rows, 0, True, 2, 1.0, False)
             for heads, rows in MODEL_SHAPES}
    cases.update({
        'rope_view_h8_m11': (256, 8, 11, 20, True, 2, 1.0, False),
        'rope_position_edges_h1_m11': (256, 1, 11, 0, True, 2, 1.0, False),
        'rope_fallback_normal': (256, 8, 11, 0, False, 0, 1.0, False),
        'rope_fallback_head128': (128, 8, 11, 0, False, 2, 1.0, False),
        'rope_fallback_rows2': (256, 8, 2, 0, False, 2, 1.0, False),
        'rope_fallback_freqscale': (256, 8, 11, 0, False, 2, 0.5, False),
        'rope_fallback_factors': (256, 8, 11, 0, False, 2, 1.0, True),
    })
    return cases


def expected_positions(name, rows, phase):
    if name == 'rope_position_edges_h1_m11':
        edges = [-7, 0, 1, 4094, 4095, 4096, 8192, -(2**31), 2**31 - 1, 123, 333]
        return [edges[(index + (0 if phase == 'initial' else 3)) % 11] for index in range(rows)]
    return [index + (0 if phase == 'initial' else 331) for index in range(rows)]


def fixture_checks(prefix, text, mode, graphs):
    ensure_other_candidates_off(text)
    data = json.loads(Path(str(prefix) + '.json').read_text())
    raw_path = Path(str(prefix) + '.bin')
    raw = raw_path.read_bytes()
    records = data['records']
    need(data.get('status') == 'tests_complete' and data.get('backend') == 'CUDA0' and
         data.get('schema_version') == 1 and data.get('gpu_alias_cases') == 0 and
         data.get('reference_budget_version') == 'rope-fp16-rounding-v1' and
         data.get('input_update_checked') is True and data.get('position_update_checked') is True and
         data.get('mode') == mode and data.get('graphs_requested') is graphs and
         data.get('case_count') == 13 and data.get('record_count') == 39 and len(records) == 39 and
         data.get('raw_bytes') == len(raw), 'RoPE fixture数量/运行设置不符')
    expected = expected_cases()
    by_case, end = {}, 0
    for item in records:
        name, phase = item['name'], item['phase']
        offset, size, elements = item['raw_offset_bytes'], item['raw_bytes'], item['elements']
        need(name in expected and phase in PHASES and item['node'] == name and
             type(item['eligible']) is bool and type(item['view_offset_bytes']) is int,
             'fixture名称/阶段/eligibility错误')
        actual = tuple(item[key] for key in ('head_dim', 'heads', 'rows', 'view_offset_bytes',
                                             'eligible', 'rope_mode', 'freq_scale', 'has_freq_factors'))
        need(actual == expected[name], f'fixture范围发生变化：{name}')
        need(type(offset) is int and type(size) is int and type(elements) is int and
             offset == end and elements == item['head_dim'] * item['heads'] * item['rows'] and
             size == 4 * elements and size > 0 and offset + size <= len(raw) and
             item['raw_dtype'] == 'little-endian-f32', 'fixture二进制范围/类型错误')
        expected_candidate = bool(mode) and item['eligible']
        need(item['mode'] == mode and item['expected_candidate'] is expected_candidate and item['finite'] is True and
             item['repeat_bit_mismatches'] == 0 and item['reference_violations'] == 0 and
             item['reference_budget_version'] == 'rope-fp16-rounding-v1' and
             item['reference_kind'] == ('flashrt_fp16' if expected_candidate else 'ggml_fp32_fastmath'),
             'fixture分派、参考或重复验证未通过')
        positions = item['position_values']
        need(isinstance(positions, list) and len(positions) == item['rows'] and
             all(type(value) is int and -(2**31) <= value < 2**31 for value in positions),
             'fixture位置向量不完整')
        need(positions == expected_positions(name, item['rows'], phase) and
             item['input_changed'] is (phase == 'input_changed') and
             item['positions_changed'] is (phase != 'initial'), 'fixture位置边界覆盖或输入更新标记不符')
        skipped = (sum(abs(pos) > 8192 for pos in positions) * item['heads'] * item['head_dim']
                   if not expected_candidate else 0)
        need(item['reference_skipped_elements'] == skipped and
             item['reference_elements'] == elements - skipped and item['reference_elements'] > 0,
             'fixture数学参考覆盖数量不符')
        need(all(type(item[key]) in (int, float) and math.isfinite(item[key]) and item[key] >= 0
                 for key in ('reference_max_abs', 'reference_rmse', 'reference_max_budget_ratio')) and
             item['reference_max_budget_ratio'] <= 1.0, 'fixture数学误差统计或误差预算无效')
        need(item['compute_calls'] == (8 if phase == 'initial' else 3),
             'fixture捕获/重放/更新次数不符')
        need(all(math.isfinite(value[0]) for value in
                 struct.iter_unpack('<f', raw[offset:offset + size])), 'fixture原始输出包含非有限值')
        case_records = by_case.setdefault(name, {})
        need(phase not in case_records, 'fixture重复阶段记录')
        case_records[phase] = item
        end += size
    need(end == len(raw) and set(by_case) == set(expected) and
         all(set(items) == set(PHASES) for items in by_case.values()),
         'fixture二进制长度或案例阶段不完整')
    for name, phases in by_case.items():
        initial, pos, inp = (phases[phase] for phase in PHASES)
        need(initial['input_fingerprint'] == pos['input_fingerprint'] != inp['input_fingerprint'] and
             initial['position_fingerprint'] != pos['position_fingerprint'] == inp['position_fingerprint'] and
             initial['position_values'] != pos['position_values'] == inp['position_values'],
             f'输入更新与位置更新没有独立执行：{name}')
        byte_outputs = [raw[item['raw_offset_bytes']:item['raw_offset_bytes'] + item['raw_bytes']]
                        for item in (initial, pos, inp)]
        need(byte_outputs[0] != byte_outputs[1] and byte_outputs[1] != byte_outputs[2],
             f'位置或输入更新后输出不变：{name}')
    hits = trace_records(text)
    if mode == 0:
        need(not hits, 'baseline意外命中新RoPE')
    else:
        for hit in hits:
            item = by_case.get(hit.get('node'), {}).get('initial')
            need(item is not None and item['eligible'], 'fallback或未知fixture意外命中新RoPE')
            need(hit.get('heads') == str(item['heads']) and hit.get('rows') == str(item['rows']) and
                 hit.get('input') == 'f32' and hit.get('output') == 'f32' and
                 hit.get('math') == 'fp16' and hit.get('layout') == 'neox' and
                 hit.get('capture') in ('0', '1'), 'fixture TRACE形状/精度/布局/capture不符')
        for name, item in expected.items():
            if not item[4]:
                continue
            matching = [hit for hit in hits if hit.get('node') == name]
            need(matching, f'支持的fixture未实际命中：{name}')
            if graphs:
                need(any(hit['capture'] == '1' for hit in matching), f'fixture未捕获新RoPE：{name}')
        if not graphs:
            need(all(hit['capture'] == '0' for hit in hits), 'fixture graphs-off意外capture')
    return data, raw_path, hits


def compatible_fixture_metadata(candidate, baseline, label):
    need([[row[key] for key in FIXTURE_FIELDS] for row in candidate['records']] ==
         [[row[key] for key in FIXTURE_FIELDS] for row in baseline['records']],
         label + '：fixture元数据不一致')


def check_all_bytes(candidate, baseline, label):
    data, raw_path = candidate
    old_data, old_path = baseline
    compatible_fixture_metadata(data, old_data, label)
    need(raw_path.read_bytes() == old_path.read_bytes(), label + '：输出原始字节不一致')
    return len(data['records'])


def compare_fixture_modes(candidate, baseline):
    data, raw_path = candidate
    old_data, old_path = baseline
    compatible_fixture_metadata(data, old_data, 'candidate/baseline')
    raw, old_raw = raw_path.read_bytes(), old_path.read_bytes()
    comparisons, fallback_records = [], 0
    for item in data['records']:
        offset, size = item['raw_offset_bytes'], item['raw_bytes']
        left, right = raw[offset:offset + size], old_raw[offset:offset + size]
        if not item['eligible']:
            need(left == right, f'fallback改变原GGML输出：{item["name"]}/{item["phase"]}')
            fallback_records += 1
        # Native FP16 arithmetic may differ from GGML FP32; report, do not turn it into an accuracy gate.
        values = [(x[0], y[0]) for x, y in zip(struct.iter_unpack('<f', left), struct.iter_unpack('<f', right))]
        diff_sq = math.fsum((x - y) ** 2 for x, y in values)
        ref_sq = math.fsum(y * y for _, y in values)
        comparisons.append(dict(name=item['name'], phase=item['phase'], eligible=item['eligible'],
                                max_abs=max(abs(x - y) for x, y in values),
                                rmse=math.sqrt(diff_sq / len(values)),
                                relative_l2=math.sqrt(diff_sq / ref_sq) if ref_sq else
                                (0.0 if diff_sq == 0 else None), binary_exact=left == right))
    need(fallback_records == 15, 'fallback阶段记录数量不符')
    return dict(fallback_phase_records_binary_exact=fallback_records, records=comparisons)


def check_model_hits(text, mode, graphs):
    ensure_other_candidates_off(text)
    hits = trace_records(text)
    if mode == 0:
        need(not hits, '模型baseline意外命中新RoPE')
        return hits
    for hit in hits:
        shape = (int(hit['heads']), int(hit['rows']))
        need(shape in MODEL_SHAPES and hit.get('input') == 'f32' and hit.get('output') == 'f32' and
             hit.get('math') == 'fp16' and hit.get('layout') == 'neox' and
             hit.get('capture') in ('0', '1'), '真实模型RoPE命中超出本次形状/布局/精度范围')
    for heads, rows in sorted(MODEL_SHAPES):
        matching = [hit for hit in hits if hit.get('heads') == str(heads) and hit.get('rows') == str(rows)]
        need(matching, f'真实模型未命中RoPE heads={heads}, rows={rows}，不能把回退当作完成')
        if graphs:
            need(any(hit['capture'] == '1' for hit in matching),
                 f'真实模型未捕获RoPE heads={heads}, rows={rows}')
    if not graphs:
        need(all(hit['capture'] == '0' for hit in hits), '模型graphs-off意外capture')
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.home() / 'Jetson-PI-Edge-flashrt-sm89')
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--gpu')
    args = ap.parse_args(argv)
    root = args.root.resolve(strict=True)
    results = root / 'results/flashrt-sm89'
    results.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='rope-rotation-', dir=results))
    report = dict(status='running', full_port_complete=False, stages={}, fixtures={}, cleanup_errors=[],
                  performance='not_measured', task_accuracy='not_evaluated', runner_sha256=sha(__file__),
                  kernel_math='FlashRT FP16 quantized input/coefficient/output rotation, adapted to GGML NEOX split-half F32 storage; no packed QKV or KV-write fusion.',
                  numerical_policy='Every candidate fixture element passes the independent FP16 analytic reference. GGML fast-math reference omits abs(position)>8192; explicit coverage counts are retained. Fallback A/B and same-mode Graph bytes exact. Baseline matches accepted actions; candidate finite/repeat/Graph consistency checked; action drift does not accept task accuracy.',
                  scope='RoPE rotation only: head_dim256, heads1/8, rows11/264/268, NEOX; patch_im2col and prior accepted kernels retained.',
                  remaining_native_components=['packed_QKV_split', 'fused_RoPE_KV_write'])
    print('本次结果目录:', out, flush=True)
    try:
        refpath = (args.reference or root / REFERENCE).resolve(strict=True)
        ref = json.loads(refpath.read_text())
        need(ref.get('status') == 'patch_im2col_integrated' and
             ref.get('model_numerics') == 'exact_for_three_cases_in_both_graph_settings' and ref.get('cleanup_errors') == [],
             '缺少已完成、数值精确且进程清理正常的patch_im2col阶段记录')
        patch = load_checked(root / PATCH_RUNNER, PATCH_RUNNER_SHA)
        p = patch.load_checked(root / PREFILL_RUNNER, PREFILL_RUNNER_SHA)
        g = p.load_checked(root / GEGLU_RUNNER, GEGLU_RUNNER_SHA)
        paths = {'runner_sha256': root / 'scripts/run-sm89-activation-stage.py',
                 'helper_sha256': root / 'run-ggml-math-shadow.py',
                 'micro_helper_sha256': root / 'scripts/run-sm89-gemm-stage.py'}
        a, h, micro = [load_checked(path, ref['helper_code_sha256'][key]) for key, path in paths.items()]
        d = load_checked(root / g.DIAGNOSTIC, g.DIAGNOSTIC_SHA)
        need(h.listener_pid() is None, '18081已占用；未停止已有进程')
        cases = {item['id']: item for item in ref['input']['cases']}
        need(set(cases) == set(g.SEQUENCE) and len(ref['input']['cases']) == 3, '参考输入案例不完整')
        targets, baseline_stages = {}, {}
        for graphs in (False, True):
            stage = ref['stages'][f'model-graphs-{"on" if graphs else "off"}-mode-1']
            need(stage['status'] == 'passed' and [item['case'] for item in stage['records']] == g.SEQUENCE,
                 '保留patch_im2col、prefill/GEGLU关闭的参考10请求不完整')
            target = {}
            for item in stage['records']:
                response, name = item['response'], item['case']
                need(g.exact(g.compare(response, target.get(name, response))), '参考基线重复不一致')
                target[name] = response
            need(set(target) == set(cases), '参考基线缺少输入案例')
            targets[(graphs, 0)] = target
            baseline_stages[graphs] = stage
        for name in cases:
            need(g.exact(g.compare(targets[(False, 0)][name], targets[(True, 0)][name])),
                 '参考基线本身Graph开关不一致')
        saved = baseline_stages[False]['environment']
        expected_flags = {'GGML_CUDA_FLASHRT': '1', 'GGML_CUDA_FLASHRT_GGML_MATH': '1',
                          'GGML_CUDA_FLASHRT_RMSNORM': '1', 'GGML_CUDA_FLASHRT_GEMM': '2',
                          'GGML_CUDA_FLASHRT_BIAS_RESIDUAL': '1',
                          'GGML_CUDA_FLASHRT_BIAS_ACTIVATION': '0', 'GGML_CUDA_FLASHRT_GEGLU': '0',
                          'GGML_CUDA_FLASHRT_PREFILL_GEMM': '0',
                          'GGML_CUDA_FLASHRT_PATCH_IM2COL': '1'}
        for stage in baseline_stages.values():
            need(all(stage['environment'].get(key) == value for key, value in expected_flags.items()),
                 '参考基线不是已验收norm/GEMM2/bias-residual/patch_im2col、prefill和GEGLU关闭的配置')
            need(COMPUTE_OVERRIDE not in stage['environment'], f'参考存在{COMPUTE_OVERRIDE}，未改变设置')
        runtime = baseline_stages[False]['inherited_runtime_environment']
        need(set(runtime) == set(g.RUNTIME_KEYS) and runtime == baseline_stages[True]['inherited_runtime_environment'] and
             all(value is None or isinstance(value, str) for value in runtime.values()), '参考运行环境缺失或Graph组之间变化')
        command = baseline_stages[False]['command']
        need(isinstance(command, list) and command and all(isinstance(value, str) for value in command) and
             command == baseline_stages[True]['command'] and
             not any(value.startswith(('--override-tensor', '-ot')) for value in command),
             '参考命令发生变化或包含额外GPU放置')
        binary = Path(command[0]).resolve(strict=True)
        need(binary.name == 'llama-server' and binary.parent.name == 'bin', '参考服务路径格式错误')
        build = binary.parent.parent
        need(build.is_relative_to(root) and Path(ref['build']['directory']).resolve() == build,
             '参考build不在当前仓库或记录不一致')
        models = [Path(item['path']).resolve(strict=True) for item in ref['models']]
        image = Path(ref['input']['image']).resolve(strict=True)
        need(len(models) == 2, '模型/mmproj记录不完整')
        print('校验固定模型与输入...', flush=True)
        need({str(path): sha(path) for path in models} == ref['model_sha256'] and
             sha(image) == ref['input']['image_sha256'], '模型或输入图像变化')
        report.update(reference=str(refpath), reference_sha256=sha(refpath), input=ref['input'], models=ref['models'],
                      model_sha256=ref['model_sha256'], helper_code_sha256=ref['helper_code_sha256'],
                      geglu_helper_sha256=GEGLU_RUNNER_SHA, prefill_helper_sha256=PREFILL_RUNNER_SHA,
                      patch_helper_sha256=PATCH_RUNNER_SHA,
                      inherited_runtime_environment=runtime,
                      build={'directory': str(build)})
        # Validate precision/environment before building or selecting a GPU.
        environment(g, saved, runtime, 'GPU-not-yet-selected', 0, False)
        before_sources = a.source_hashes(root, micro)
        cmake = shutil.which('cmake') or str(Path.home() / '.local/bin/cmake')
        g.run([cmake, '-S', root, '-B', build, '-DGGML_CUDA=ON', '-DGGML_CUDA_FLASHRT=ON',
               '-DCMAKE_CUDA_ARCHITECTURES=89', '-DLLAMA_BUILD_TESTS=ON'], out / 'configure.log')
        g.run([cmake, '--build', build, '--target', 'llama-server', EXE, '-j', '8'], out / 'build.log')
        need(before_sources == a.source_hashes(root, micro), '构建期间源码变化')

        def artifact_hashes():
            result = a.artifact_hashes(build)
            result['bin/' + EXE] = sha(build / 'bin' / EXE)
            return result

        manifest = dict(root=str(root), build=str(build), sources=a.source_hashes(root, micro),
                        artifacts=artifact_hashes(), cache_sha256=sha(build / 'CMakeCache.txt'))
        need(all(path in manifest['sources'] for path in NEW_SOURCES), 'manifest未包含全部新kernel、guard、接口、测试和runner源码')
        save(out / 'build-manifest.json', manifest)
        report['build_manifest'] = str(out / 'build-manifest.json')
        host_text = g.run([build / 'bin' / EXE, '--host-only'], out / 'host.log')
        print(host_text, end='', flush=True)
        host_match = re.search(r'(\d+) RoPE metadata/address contract checks passed', host_text)
        need(host_match and int(host_match[1]) > 0, '缺少主机契约检查通过记录')
        report['host_contract_checks'] = int(host_match[1])
        if args.gpu:
            snapshot = a.wait_for_idle_gpu(args.gpu, out / 'idle', micro)
            selection = {'selected_uuid': snapshot['uuid'], 'confirmed_snapshot': snapshot}
        else:
            selection = d.select_gpu(ref['gpu_selection']['selected_uuid'], out, a, micro)
        gpu = selection['selected_uuid']
        report['gpu_selection'] = selection

        def verify():
            need(a.source_hashes(root, micro) == manifest['sources'] and artifact_hashes() == manifest['artifacts'] and
                 sha(build / 'CMakeCache.txt') == manifest['cache_sha256'], '运行期间源码或二进制变化')
            need(sha(refpath) == report['reference_sha256'] and sha(image) == ref['input']['image_sha256'],
                 '运行期间参考或输入变化')

        fixtures = {}
        for graphs in (False, True):
            for mode in (0, 1):
                label = f'fixtures-graphs-{"on" if graphs else "off"}-mode-{mode}'
                verify()
                a.wait_for_idle_gpu(gpu, out / (label + '-idle'), micro)
                prefix = out / label
                fixture_command = [str(value) for value in
                                   (build / 'bin' / EXE, '--mode', mode, '--graphs',
                                    'on' if graphs else 'off', '--output-prefix', prefix)]
                fixture_env = environment(g, saved, runtime, gpu, mode, graphs)
                report['fixtures'][label] = {
                    'status': 'running', 'result': str(prefix) + '.json',
                    'command': fixture_command, 'gpu_uuid': gpu,
                    'environment': {key: value for key, value in fixture_env.items()
                                    if key.startswith(('GGML_', 'CUDA_', 'PI_MODEL'))},
                    'log': str(out / (label + '.log'))}
                save(out / 'summary.json', report)
                text = g.run(fixture_command, out / (label + '.log'), fixture_env)
                data, raw_path, hits = fixture_checks(prefix, text, mode, graphs)
                need(data.get('host_checks') == report['host_contract_checks'], 'fixture与独立主机契约检查数量不符')
                ab = compare_fixture_modes((data, raw_path), fixtures[(graphs, 0)]) if mode == 1 else None
                if graphs:
                    check_all_bytes((data, raw_path), fixtures[(False, mode)], '同模式Graph开关')
                fixtures[(graphs, mode)] = (data, raw_path)
                report['fixtures'][label].update(status='passed', trace_records=len(hits),
                                                 capture_records=sum(hit.get('capture') == '1' for hit in hits),
                                                 ab_comparison=ab,
                                                 reference_elements=sum(row['reference_elements'] for row in data['records']),
                                                 reference_violations=0)
                print(f'{label}: 13 cases/39 records，全元素数学参考通过；TRACE={len(hits)}', flush=True)
                save(out / 'summary.json', report)
        report['fixture_correctness'] = 'all_elements_independent_reference_fallback_and_graph_bytes_exact_with_input_and_position_updates'
        for graphs in (False, True):
            for mode in (0, 1):
                label = f'model-graphs-{"on" if graphs else "off"}-mode-{mode}'
                verify()
                a.wait_for_idle_gpu(gpu, out / (label + '-idle'), micro)
                need(h.listener_pid() is None, '18081被占用；未停止已有进程')
                env = environment(g, saved, runtime, gpu, mode, graphs)
                stage = dict(status='running', records=[], command=command,
                             environment={key: value for key, value in env.items()
                                          if key.startswith(('GGML_', 'CUDA_', 'PI_MODEL'))},
                             inherited_runtime_environment={key: env.get(key) for key in g.RUNTIME_KEYS})
                report['stages'][label] = stage
                logpath = out / (label + '.log')
                stage['log'] = str(logpath)
                stage['gpu_uuid'] = gpu
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                repeated = {}
                print('阶段:', label, flush=True)
                with logpath.open('w') as log:
                    child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    stage['owned_pid'] = child.pid
                    try:
                        h.wait_ready(child, opener)
                        for index, name in enumerate(g.SEQUENCE, 1):
                            need(child.poll() is None and h.listener_pid() == child.pid, '服务身份变化')
                            h.verify_server_identity(child.pid, binary, *models)
                            need(h.request(opener, 'POST', '/foreground/reset').get('reset') is True, 'reset失败')
                            need(h.request(opener, 'POST', '/foreground/image', {'path': str(image)}, timeout=30).get('loaded') is True,
                                 'image失败')
                            state = h.request(opener, 'PUT', '/foreground/state', {'state': cases[name]['state']})
                            need(state.get('loaded') is True and state.get('state_dim') == 32, 'state失败')
                            response = h.request(opener, 'POST', '/foreground/infer', {'text': cases[name]['text']}, timeout=120)
                            record = dict(request_index=index, case=name, response=a.safe_json(response))
                            stage['records'].append(record)
                            save(out / 'summary.json', report)
                            diff = g.compare(response, targets[(graphs, 0)][name])
                            record['vs_accepted_baseline'] = diff
                            need(diff['action']['exact'], '初始噪声变化')
                            need(g.exact(g.compare(response, repeated.get(name, response))), '同输入重复输出变化')
                            if mode == 0:
                                need(g.exact(diff), 'RoPE关闭的基线与已验收patch_im2col动作不一致')
                            repeated[name] = response
                            print(f'{label}: {index}/10 {name}, max_abs={diff["action_final_raw"]["max_abs"]:.9g}', flush=True)
                    except BaseException as error:
                        stage.update(status='failed', error=f'{type(error).__name__}: {error}')
                        raise
                    finally:
                        try:
                            h.stop_child(child)
                        except BaseException as error:
                            report['cleanup_errors'].append(f'{label} owned PID {child.pid}: {error}')
                        save(out / 'summary.json', report)
                need(not report['cleanup_errors'], '自有服务清理未完成')
                hits = check_model_hits(logpath.read_text(errors='replace'), mode, graphs)
                stage.update(status='passed', trace_records=len(hits),
                             capture_records=sum(hit.get('capture') == '1' for hit in hits),
                             output_types=sorted({hit['output'] for hit in hits}))
                print(f'{label} 通过：动作有限且重复一致，RoPE TRACE={len(hits)}', flush=True)
                targets[(graphs, mode)] = repeated
                save(out / 'summary.json', report)
        verify()
        need({str(path): sha(path) for path in models} == ref['model_sha256'], '测试期间模型文件变化')
        report['model_comparisons'] = {name: g.compare(targets[(False, 1)][name], targets[(False, 0)][name]) for name in cases}
        report['baseline_graph_comparisons'] = {name: g.compare(targets[(True, 0)][name], targets[(False, 0)][name]) for name in cases}
        report['model_graph_comparisons'] = {name: g.compare(targets[(True, 1)][name], targets[(False, 1)][name]) for name in cases}
        need(all(g.exact(value) for value in report['baseline_graph_comparisons'].values()), 'RoPE关闭基线Graph开关不一致')
        need(all(g.exact(value) for value in report['model_graph_comparisons'].values()), 'RoPE候选模型Graph开关不一致')
        same = all(g.exact(value) for value in report['model_comparisons'].values())
        report.update(status='rope_rotation_integrated',
                      model_numerics='exact_for_three_cases_in_both_graph_settings' if same
                      else 'drift_recorded_task_accuracy_not_accepted')
        print('\ncase,max_abs,RMSE,relative_L2', flush=True)
        for name, diff in report['model_comparisons'].items():
            value = diff['action_final_raw']
            print(f'{name},{value["max_abs"]:.9g},{value["rmse"]:.9g},{value["relative_l2"]}', flush=True)
        return 0
    except BaseException as error:
        report.update(status='incomplete', error=f'{type(error).__name__}: {error}')
        for stage in list(report['stages'].values()) + list(report['fixtures'].values()):
            if stage.get('status') == 'running':
                stage.update(status='failed', error=report['error'])
        print(report['error'], flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        save(out / 'summary.json', report)
        print('status:', report['status'], flush=True)
        if 'model_numerics' in report:
            print('model_numerics:', report['model_numerics'], flush=True)
        print('已保存:', out / 'summary.json', flush=True)
        print('本次接入FlashRT FP16 RoPE旋转的GGML NEOX布局适配；QKV拆分与KV写入融合待完成。prefill GEMM和GEGLU保持关闭；未验收任务精度或性能收益，完整kernel移植仍未完成。', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
