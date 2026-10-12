#!/usr/bin/env python3
"""Verify FlashRT fp16_nn prefill GEMM on SM89; preserve CPU placement and accepted kernels."""
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

REFERENCE = 'results/flashrt-sm89/geglu-port-o5qr0oof/summary.json'
GEGLU_RUNNER = 'scripts/run-sm89-geglu-port.py'
GEGLU_RUNNER_SHA = 'e963bc41c20dfc70abc493de5c98dfd2040cb57a7c1903b0d23f35176686d405'
FEATURE = 'GGML_CUDA_FLASHRT_PREFILL_GEMM'
COMPUTE_OVERRIDE = 'GGML_CUDA_CUBLAS_COMPUTE_TYPE'
TRACE = 'flashrt_prefill_gemm:'
EXE = 'test-flashrt-prefill-gemm'
VISION_SHAPES = {(1152, 1152, 256), (1152, 4304, 256),
                 (4304, 1152, 256), (1152, 2048, 256)}
ENCODER_SHAPES = {(k, n, m) for m in (264, 268) for k, n in
                  ((2048, 2048), (2048, 256), (2048, 16384), (16384, 2048))}
MODEL_SHAPES = VISION_SHAPES | ENCODER_SHAPES
FIXTURE_FIELDS = ('name', 'node', 'phase', 'k', 'n', 'm', 'input_type', 'precision',
                  'eligible', 'elements', 'raw_dtype', 'raw_offset_bytes', 'raw_bytes')


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


def shape_of(hit):
    return tuple(int(hit[key]) for key in ('K', 'N', 'M'))


def environment(g, saved, runtime, gpu, mode, graphs):
    # Do not silently erase an explicit user or saved baseline precision override.
    need(COMPUTE_OVERRIDE not in saved and COMPUTE_OVERRIDE not in os.environ,
         f'{COMPUTE_OVERRIDE} 已显式设置，与本次固定 native FP16 路径不兼容；保留原设置并停止')
    env = g.environment(saved, runtime, gpu, 0, graphs, trace=True)
    env.update({FEATURE: str(mode), 'GGML_CUDA_FLASHRT_GEGLU': '0',
                'GGML_CUDA_FLASHRT_BIAS_ACTIVATION': '0'})
    return env


def fixture_checks(prefix, text, mode, graphs):
    data = json.loads(Path(str(prefix) + '.json').read_text())
    raw_path = Path(str(prefix) + '.bin')
    length = raw_path.stat().st_size
    records = data['records']
    need(data.get('status') == 'tests_complete' and data.get('input_update_checked') is True and
         data.get('mode') == mode and data.get('graphs_requested') is graphs and
         data.get('case_count') == 17 and data.get('record_count') == 34 and len(records) == 34 and
         data.get('raw_bytes') == length, 'prefill GEMM fixture数量/运行设置不符')
    end, pairs, initial = 0, set(), {}
    for item in records:
        offset, size, elements = item['raw_offset_bytes'], item['raw_bytes'], item['elements']
        need(type(offset) is int and type(size) is int and type(elements) is int and offset == end and
             size == 4 * elements and elements == item['n'] * item['m'] and size > 0 and
             offset + size <= length, 'fixture二进制范围错误')
        need(item.get('raw_dtype') == 'little-endian-f32' and item['name'] == item['node'] and
             item['input_type'] in ('f32', 'f16') and item['phase'] in ('initial', 'changed'),
             'fixture布局/类型/阶段标记错误')
        need(item['repeat_bit_mismatches'] == 0 and item['finite'] is True and
             item['reference_samples'] == 96, 'fixture有限值、重放或96点参考记录不完整')
        gate_applied = bool(mode) and item['eligible']
        need(item['reference_gate_applied'] is gate_applied and
             type(item['reference_bound_failures']) is int and 0 <= item['reference_bound_failures'] <= 96,
             'fixture参考检查策略或真实失败计数不符')
        if gate_applied:
            need(item['reference_bound_failures'] == 0, '新kernel的96点独立数学参考未通过')
        need(item['tolerance_atol'] == 0.004 and item['tolerance_rtol'] == 0.002,
             'fixture参考误差门槛被改变')
        need(item['compute_calls'] == (8 if item['phase'] == 'initial' else 3),
             'fixture重复/输入更新次数不符')
        need(item['expected_candidate'] is (bool(mode) and item['eligible']), 'fixture期望分派不符')
        pairs.add((item['name'], item['phase']))
        if item['phase'] == 'initial':
            initial[item['name']] = item
        end += size
    need(end == length and len(pairs) == 34 and len(initial) == 17 and
         pairs == {(name, phase) for name in initial for phase in ('initial', 'changed')},
         'fixture二进制长度或案例阶段不完整')
    expected = {(k, n, m, 'f32', 'default', True) for k, n, m in MODEL_SHAPES}
    expected |= {(1152, 1152, 256, 'f16', 'default', True),
                 (2048, 256, 2, 'f32', 'default', False),
                 (2048, 256, 11, 'f32', 'default', False),
                 (2048, 256, 265, 'f32', 'default', False),
                 (1152, 1152, 256, 'f32', 'f32', False)}
    need({(r['k'], r['n'], r['m'], r['input_type'], r['precision'], r['eligible'])
          for r in initial.values()} == expected, 'fixture实际形状范围不符')
    for item in records:
        start = initial[item['name']]
        need(all(item[key] == start[key] for key in
                 ('k', 'n', 'm', 'input_type', 'precision', 'eligible', 'elements')),
             '更新输入时fixture元数据变化')
    hits = trace_records(text)
    if mode == 0:
        need(not hits, 'baseline意外命中新prefill GEMM')
    else:
        for hit in hits:
            item = initial.get(hit.get('node'))
            need(item is not None and item['eligible'], '不支持的fixture意外命中新kernel')
            need(shape_of(hit) == (item['k'], item['n'], item['m']) and
                 hit.get('input') == item['input_type'] and hit.get('core_output') == 'f16',
                 'fixture TRACE的形状或dtype与测试不符')
        for name, item in initial.items():
            if not item['eligible']:
                continue
            matching = [hit for hit in hits if hit.get('node') == name]
            need(matching, f'支持的fixture未实际命中：{name}')
            if graphs:
                need(any(hit.get('capture') == '1' for hit in matching), f'fixture未捕获新kernel：{name}')
        need(all(hit.get('capture') in ('0', '1') for hit in hits), 'capture标记缺失')
        if not graphs:
            need(all(hit['capture'] == '0' for hit in hits), 'graphs-off意外capture')
    return data, raw_path, hits


def check_fallback_bytes(candidate, baseline):
    data, raw_path = candidate
    old_data, old_path = baseline
    need([[row[key] for key in FIXTURE_FIELDS] for row in data['records']] ==
         [[row[key] for key in FIXTURE_FIELDS] for row in old_data['records']],
         'candidate/baseline fixture元数据不一致')
    checked = 0
    with raw_path.open('rb') as current, old_path.open('rb') as reference:
        for item in data['records']:
            if item['eligible']:
                continue
            current.seek(item['raw_offset_bytes'])
            reference.seek(item['raw_offset_bytes'])
            need(current.read(item['raw_bytes']) == reference.read(item['raw_bytes']),
                 f'fallback未保持原GGML输出：{item["name"]}/{item["phase"]}')
            checked += 1
    need(checked == 8, 'fallback阶段记录数量不符')
    return checked


def check_model_hits(text, mode, graphs):
    hits = trace_records(text)
    if mode == 0:
        need(not hits, '模型baseline意外命中新prefill GEMM')
        return hits
    for hit in hits:
        shape = shape_of(hit)
        need(shape in MODEL_SHAPES and hit.get('input') == 'f32' and hit.get('core_output') == 'f16',
             '真实模型命中超出本次12形状或dtype范围')
        expected_domain = 'vision' if shape in VISION_SHAPES else 'encoder'
        need(hit.get('domain') == expected_domain, '模型TRACE区域标记不符')
        need(hit.get('capture') in ('0', '1'), '模型capture标记缺失')
    for shape in sorted(MODEL_SHAPES):
        matching = [hit for hit in hits if shape_of(hit) == shape]
        need(matching, f'真实模型未命中prefill GEMM K,N,M={shape}')
        if graphs:
            need(any(hit['capture'] == '1' for hit in matching), f'模型未capture形状 K,N,M={shape}')
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
    out = Path(tempfile.mkdtemp(prefix='prefill-gemm-', dir=results))
    report = dict(status='running', full_port_complete=False, stages={}, fixtures={}, cleanup_errors=[],
                  performance='not_measured', runner_sha256=sha(__file__),
                  kernel_math='FlashRT fp16_nn: F16 operands, FP32 compute, F16 core output widened to GGML F32; F32 input rounded RN to F16',
                  numerical_policy='96 independent double dot products per fixture phase. Fixed error gate applies only to candidate eligible kernels; original GGML and fallback errors are recorded. All outputs must be finite, repeats and fallback bytes exact; model drift is never accepted as task accuracy.',
                  model_shapes=[dict(k=k, n=n, m=m) for k, n, m in sorted(MODEL_SHAPES)])
    print('本次结果目录:', out, flush=True)
    try:
        refpath = (args.reference or root / REFERENCE).resolve(strict=True)
        ref = json.loads(refpath.read_text())
        need(ref.get('status') == 'geglu_kernel_integrated' and ref.get('cleanup_errors') == [],
             '缺少已完成且进程清理正常的GEGLU阶段记录')
        g = load_checked(root / GEGLU_RUNNER, GEGLU_RUNNER_SHA)
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
                 'GEGLU关闭的参考基线10请求不完整')
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
                          'GGML_CUDA_FLASHRT_BIAS_ACTIVATION': '0', 'GGML_CUDA_FLASHRT_GEGLU': '0'}
        for stage in baseline_stages.values():
            need(all(stage['environment'].get(key) == value for key, value in expected_flags.items()),
                 '参考基线不是已验收norm/GEMM2/bias-residual、GEGLU关闭的配置')
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
        need(build.is_relative_to(root), '参考build不在当前仓库')
        models = [Path(item['path']).resolve(strict=True) for item in ref['models']]
        image = Path(ref['input']['image']).resolve(strict=True)
        need(len(models) == 2, '模型/mmproj记录不完整')
        print('校验固定模型与输入...', flush=True)
        need({str(path): sha(path) for path in models} == ref['model_sha256'] and
             sha(image) == ref['input']['image_sha256'], '模型或输入图像变化')
        report.update(reference=str(refpath), reference_sha256=sha(refpath), input=ref['input'], models=ref['models'],
                      model_sha256=ref['model_sha256'], helper_code_sha256=ref['helper_code_sha256'],
                      geglu_helper_sha256=GEGLU_RUNNER_SHA, inherited_runtime_environment=runtime,
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
        need('ggml/src/ggml-cuda/flashrt-prefill-gemm.cu' in manifest['sources'] and
             'tests/test-flashrt-prefill-gemm.cpp' in manifest['sources'], 'manifest未包含新kernel或测试源码')
        save(out / 'build-manifest.json', manifest)
        report['build_manifest'] = str(out / 'build-manifest.json')
        host_text = g.run([build / 'bin' / EXE, '--host-only'], out / 'host.log')
        print(host_text, end='', flush=True)
        host_match = re.search(r'(\d+) prefill GEMM metadata/address contract checks passed', host_text)
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
                fallback_checked = check_fallback_bytes((data, raw_path), fixtures[(graphs, 0)]) if mode == 1 else 0
                if graphs:
                    old_data, old_path = fixtures[(False, mode)]
                    need([[row[key] for key in FIXTURE_FIELDS] for row in data['records']] ==
                         [[row[key] for key in FIXTURE_FIELDS] for row in old_data['records']] and
                         raw_path.read_bytes() == old_path.read_bytes(), '同一prefill GEMM模式Graph开关输出不一致')
                fixtures[(graphs, mode)] = (data, raw_path)
                report['fixtures'][label].update(status='passed', trace_records=len(hits),
                                                 capture_records=sum(hit.get('capture') == '1' for hit in hits),
                                                 fallback_phase_records_binary_exact=fallback_checked,
                                                 reference_bound_failures_recorded=sum(row['reference_bound_failures'] for row in data['records']),
                                                 reference_gated_phase_records=sum(row['reference_gate_applied'] for row in data['records']))
                print(label + ': 17 cases，全部输出有限/重复与输入更新通过；新kernel检查96点参考，回退检查原始输出', flush=True)
                save(out / 'summary.json', report)
        report['fixture_correctness'] = 'candidate_sampled_reference_all_finite_repeat_update_fallback_and_graph_bytes_passed'
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
                            if mode == 0:
                                need(g.exact(diff), '新增prefill GEMM关闭时已验收基线变化')
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
                             hit_shapes=[list(shape) for shape in sorted({shape_of(hit) for hit in hits})])
                targets[(graphs, mode)] = repeated
                save(out / 'summary.json', report)
        verify()
        need({str(path): sha(path) for path in models} == ref['model_sha256'], '测试期间模型文件变化')
        report['model_comparisons'] = {name: g.compare(targets[(False, 1)][name], targets[(False, 0)][name]) for name in cases}
        report['model_graph_comparisons'] = {name: g.compare(targets[(True, 1)][name], targets[(False, 1)][name]) for name in cases}
        need(all(g.exact(value) for value in report['model_graph_comparisons'].values()), 'prefill GEMM模型Graph开关不一致')
        same = all(g.exact(value) for value in report['model_comparisons'].values())
        report.update(status='prefill_gemm_integrated',
                      model_numerics='exact_for_three_cases' if same else 'drift_recorded_task_accuracy_not_accepted')
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
        print('本次接入FlashRT FP16 prefill/vision GEMM；GEGLU保持关闭。未验收任务精度或性能收益，完整移植仍未完成。', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
