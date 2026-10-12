#!/usr/bin/env python3
"""Verify FlashRT patch_im2col adapted to GGML CHW layout; require exact copy semantics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import tempfile
import types
import urllib.request

REFERENCE = 'results/flashrt-sm89/prefill-gemm-obwek26o/summary.json'
PREFILL_RUNNER = 'scripts/run-sm89-prefill-gemm.py'
PREFILL_RUNNER_SHA = '1f655e029ab0dc513a4a90c717c03fd1d2884330cf2cca08a977403fc579d785'
GEGLU_RUNNER = 'scripts/run-sm89-geglu-port.py'
GEGLU_RUNNER_SHA = 'e963bc41c20dfc70abc493de5c98dfd2040cb57a7c1903b0d23f35176686d405'
FEATURE = 'GGML_CUDA_FLASHRT_PATCH_IM2COL'
COMPUTE_OVERRIDE = 'GGML_CUDA_CUBLAS_COMPUTE_TYPE'
TRACE = 'flashrt_patch_im2col:'
EXE = 'test-flashrt-patch-im2col'
FIXTURE_SHAPE_FIELDS = ('image_width', 'image_height', 'patch_width', 'patch_height',
                        'stride_x', 'stride_y', 'pad_x', 'pad_y', 'batch', 'output_type',
                        'view_offset_bytes', 'eligible')
FIXTURE_FIELDS = ('name', 'node', 'phase', *FIXTURE_SHAPE_FIELDS,
                  'elements', 'raw_dtype', 'raw_offset_bytes', 'raw_bytes')
NEW_SOURCES = ('ggml/src/ggml-cuda/flashrt-patch-im2col.cu',
               'ggml/src/ggml-cuda/flashrt-patch-im2col.cuh',
               'ggml/src/ggml-cuda/flashrt-patch-im2col-guard.h',
               'tests/test-flashrt-patch-im2col.cpp', 'scripts/run-sm89-patch-im2col.py')


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
    # Preserve the saved precision setting; do not conflate GEMM changes with a copy kernel.
    need(COMPUTE_OVERRIDE not in saved and COMPUTE_OVERRIDE not in os.environ,
         f'{COMPUTE_OVERRIDE} 已显式设置，与保存的基线不同；保留原设置并停止')
    env = g.environment(saved, runtime, gpu, 0, graphs, trace=True)
    env.update({FEATURE: str(mode), 'GGML_CUDA_FLASHRT_PREFILL_GEMM': '0',
                'GGML_CUDA_FLASHRT_GEGLU': '0', 'GGML_CUDA_FLASHRT_BIAS_ACTIVATION': '0'})
    return env


def ensure_other_candidates_off(text):
    need('flashrt_prefill_gemm:' not in text and 'flashrt_geglu:' not in text and
         'flashrt_bias_activation:' not in text,
         '意外命中 prefill GEMM、GEGLU 或 bias/activation；本次只切换 patch_im2col')


def expected_cases():
    result = {}
    # width, height, patch width/height, stride x/y, pad x/y, batch, output dtype, view, eligible.
    for batch in (1, 2):
        for dtype in ('f32', 'f16'):
            result[f'patch_b{batch}_{dtype}'] = (224, 224, 14, 14, 14, 14, 0, 0, batch, dtype, False, True)
    for dtype in ('f32', 'f16'):
        result[f'patch_view_b1_{dtype}'] = (224, 224, 14, 14, 14, 14, 0, 0, 1, dtype, True, True)
    result.update({
        'patch_fallback_image238_f32': (238, 224, 14, 14, 14, 14, 0, 0, 1, 'f32', False, False),
        'patch_fallback_patch16_f32': (224, 224, 16, 16, 16, 16, 0, 0, 1, 'f32', False, False),
        'patch_fallback_pad1_f32': (224, 224, 14, 14, 14, 14, 1, 1, 1, 'f32', False, False),
        'patch_fallback_stride7_f32': (224, 224, 14, 14, 7, 7, 0, 0, 1, 'f32', False, False),
    })
    return result


def fixture_checks(prefix, text, mode, graphs):
    ensure_other_candidates_off(text)
    data = json.loads(Path(str(prefix) + '.json').read_text())
    raw_path = Path(str(prefix) + '.bin')
    raw = raw_path.read_bytes()
    records = data['records']
    need(data.get('status') == 'tests_complete' and data.get('backend') == 'CUDA0' and
         data.get('input_update_checked') is True and data.get('gpu_alias_cases') == 0 and
         data.get('reference_scope') == 'all_output_elements_both_phases_every_compute' and
         data.get('mode') == mode and data.get('graphs_requested') is graphs and
         data.get('case_count') == 10 and data.get('record_count') == 20 and len(records) == 20 and
         data.get('raw_bytes') == len(raw), 'patch_im2col fixture数量/运行设置不符')
    end, pairs, initial, changed = 0, set(), {}, {}
    expected = expected_cases()
    for item in records:
        offset, size, elements = item['raw_offset_bytes'], item['raw_bytes'], item['elements']
        dtype = item['output_type']
        need(dtype in ('f32', 'f16'), 'fixture输出类型不符')
        need(type(offset) is int and type(size) is int and type(elements) is int and offset == end and
             size == (4 if dtype == 'f32' else 2) * elements and size > 0 and
             offset + size <= len(raw), 'fixture二进制范围错误')
        need(item.get('raw_dtype') == 'little-endian-' + dtype and item['name'] == item['node'] and
             item['phase'] in ('initial', 'changed'), 'fixture布局/阶段标记错误')
        name = item['name']
        need(name in expected and type(item['view_offset_bytes']) is int and item['view_offset_bytes'] >= 0 and
             type(item['eligible']) is bool, 'fixture名字/view/eligibility错误')
        actual = tuple(item[key] for key in FIXTURE_SHAPE_FIELDS[:10]) + (item['view_offset_bytes'] > 0, item['eligible'])
        need(actual == expected[name], f'fixture范围发生变化：{name}')
        ow = (item['image_width'] + 2 * item['pad_x'] - item['patch_width']) // item['stride_x'] + 1
        oh = (item['image_height'] + 2 * item['pad_y'] - item['patch_height']) // item['stride_y'] + 1
        need(elements == ow * oh * 3 * item['patch_width'] * item['patch_height'] * item['batch'],
             'fixture元素数量与卷积几何不符')
        need(item['finite'] is True and item['repeat_bit_mismatches'] == 0 and
             item['reference_bit_mismatches'] == 0 and item['reference_elements'] == elements,
             'fixture必须全部元素通过独立CPU坐标参考和重复字节检查')
        need(item['compute_calls'] == (8 if item['phase'] == 'initial' else 3),
             'fixture重复/输入更新次数不符')
        need(item['expected_candidate'] is (bool(mode) and item['eligible']), 'fixture期望分派记录不符')
        pairs.add((name, item['phase']))
        (initial if item['phase'] == 'initial' else changed)[name] = item
        end += size
    need(end == len(raw) and len(pairs) == 20 and set(initial) == set(changed) == set(expected) and
         pairs == {(name, phase) for name in expected for phase in ('initial', 'changed')},
         'fixture二进制长度或案例阶段不完整')
    for name, start in initial.items():
        update = changed[name]
        need(all(start[key] == update[key] for key in (*FIXTURE_SHAPE_FIELDS, 'elements')),
             '更新输入时fixture元数据变化')
        old = raw[start['raw_offset_bytes']:start['raw_offset_bytes'] + start['raw_bytes']]
        new = raw[update['raw_offset_bytes']:update['raw_offset_bytes'] + update['raw_bytes']]
        need(old != new, f'更换输入后输出没有变化：{name}')
    hits = trace_records(text)
    if mode == 0:
        need(not hits, 'baseline意外命中新patch_im2col')
    else:
        for hit in hits:
            item = initial.get(hit.get('node'))
            need(item is not None and item['eligible'], 'fallback或未知fixture意外命中新kernel')
            need(hit.get('input') == 'f32' and hit.get('output') == item['output_type'] and
                 hit.get('batch') == str(item['batch']) and hit.get('width') == '224' and
                 hit.get('height') == '224' and hit.get('patch') == '14' and
                 hit.get('capture') in ('0', '1'), 'fixture TRACE形状/类型/capture不符')
        for name, item in initial.items():
            if not item['eligible']:
                continue
            matching = [hit for hit in hits if hit.get('node') == name]
            need(matching, f'支持的fixture未实际命中：{name}')
            if graphs:
                need(any(hit['capture'] == '1' for hit in matching), f'fixture未捕获新kernel：{name}')
        if not graphs:
            need(all(hit['capture'] == '0' for hit in hits), 'graphs-off意外capture')
    return data, raw_path, hits


def check_all_bytes(candidate, baseline, label):
    data, raw_path = candidate
    old_data, old_path = baseline
    need([[row[key] for key in FIXTURE_FIELDS] for row in data['records']] ==
         [[row[key] for key in FIXTURE_FIELDS] for row in old_data['records']],
         label + '：fixture元数据不一致')
    need(raw_path.read_bytes() == old_path.read_bytes(), label + '：输出没有保持全部原始字节一致')
    return len(data['records'])


def check_model_hits(text, mode, graphs):
    ensure_other_candidates_off(text)
    hits = trace_records(text)
    if mode == 0:
        need(not hits, '模型baseline意外命中新patch_im2col')
        return hits
    need(hits, '真实模型未命中patch_im2col，不能把全路径回退当作完成')
    for hit in hits:
        need(hit.get('input') == 'f32' and hit.get('output') == 'f32' and
             hit.get('batch') == '1' and hit.get('width') == '224' and
             hit.get('height') == '224' and hit.get('patch') == '14' and
             hit.get('capture') in ('0', '1'), '真实模型TRACE超出224x224单图F32范围')
    if graphs:
        need(any(hit['capture'] == '1' for hit in hits), '真实模型没有capture新patch_im2col')
    else:
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
    out = Path(tempfile.mkdtemp(prefix='patch-im2col-', dir=results))
    report = dict(status='running', full_port_complete=False, stages={}, fixtures={}, cleanup_errors=[],
                  performance='not_measured', runner_sha256=sha(__file__),
                  kernel_math='FlashRT patch_im2col: GGML CHW F32 input, c/kh/kw patch feature order, F32 copy or F16 conversion; model F32 output.',
                  numerical_policy='All fixture elements must match an independent CPU coordinate reference bitwise; A/B and Graph settings match all serialized bytes. All model action keys must exactly match saved mode-0 baseline.',
                  scope='224x224 RGB input; 14x14 patches, stride14, no padding, dilation1; fixtures B1/B2 and contiguous input views; real pi0 B1 F32')
    print('本次结果目录:', out, flush=True)
    try:
        refpath = (args.reference or root / REFERENCE).resolve(strict=True)
        ref = json.loads(refpath.read_text())
        need(ref.get('status') == 'prefill_gemm_integrated' and ref.get('cleanup_errors') == [],
             '缺少已完成且进程清理正常的prefill阶段记录')
        p = load_checked(root / PREFILL_RUNNER, PREFILL_RUNNER_SHA)
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
            stage = ref['stages'][f'model-graphs-{"on" if graphs else "off"}-mode-0']
            need(stage['status'] == 'passed' and [item['case'] for item in stage['records']] == g.SEQUENCE,
                 'prefill/GEGLU关闭的参考基线10请求不完整')
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
                          'GGML_CUDA_FLASHRT_PREFILL_GEMM': '0'}
        for stage in baseline_stages.values():
            need(all(stage['environment'].get(key) == value for key, value in expected_flags.items()),
                 '参考基线不是已验收norm/GEMM2/bias-residual、prefill和GEGLU关闭的配置')
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
        host_match = re.search(r'(\d+) patch IM2COL metadata/address contract checks passed', host_text)
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
                report['fixtures'][label] = {'status': 'running', 'result': str(prefix) + '.json'}
                save(out / 'summary.json', report)
                text = g.run([build / 'bin' / EXE, '--mode', mode, '--graphs', 'on' if graphs else 'off',
                              '--output-prefix', prefix], out / (label + '.log'),
                             environment(g, saved, runtime, gpu, mode, graphs))
                data, raw_path, hits = fixture_checks(prefix, text, mode, graphs)
                need(data.get('host_checks') == report['host_contract_checks'], 'fixture与独立主机契约检查数量不符')
                binary_checked = check_all_bytes((data, raw_path), fixtures[(graphs, 0)], 'candidate/baseline') if mode == 1 else 0
                if graphs:
                    check_all_bytes((data, raw_path), fixtures[(False, mode)], '同模式Graph开关')
                fixtures[(graphs, mode)] = (data, raw_path)
                report['fixtures'][label].update(status='passed', trace_records=len(hits),
                                                 capture_records=sum(hit.get('capture') == '1' for hit in hits),
                                                 ab_phase_records_binary_exact=binary_checked,
                                                 reference_elements=sum(row['reference_elements'] for row in data['records']),
                                                 reference_bit_mismatches=0)
                print(f'{label}: 10 cases/20 records，全元素CPU参考binary exact；TRACE={len(hits)}', flush=True)
                save(out / 'summary.json', report)
        report['fixture_correctness'] = 'all_elements_cpu_reference_ab_and_graph_bytes_exact_with_input_updates'
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
                            need(g.exact(diff), 'patch_im2col纯搬运路径的模型动作与已验收基线不一致')
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
                print(f'{label} 通过：动作exact，patch_im2col TRACE={len(hits)}', flush=True)
                targets[(graphs, mode)] = repeated
                save(out / 'summary.json', report)
        verify()
        need({str(path): sha(path) for path in models} == ref['model_sha256'], '测试期间模型文件变化')
        report['model_comparisons'] = {name: g.compare(targets[(False, 1)][name], targets[(False, 0)][name]) for name in cases}
        report['model_graph_comparisons'] = {name: g.compare(targets[(True, 1)][name], targets[(False, 1)][name]) for name in cases}
        need(all(g.exact(value) for value in report['model_graph_comparisons'].values()), 'patch_im2col模型Graph开关不一致')
        need(all(g.exact(value) for value in report['model_comparisons'].values()), 'patch_im2col候选与基线动作不一致')
        report.update(status='patch_im2col_integrated',
                      model_numerics='exact_for_three_cases_in_both_graph_settings')
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
        print('本次验收FlashRT patch_im2col的GGML布局/类型适配；prefill GEMM和GEGLU保持关闭。未测试性能收益，完整kernel移植仍未完成。', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
