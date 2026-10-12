#!/usr/bin/env python3
"""Build and exercise the FlashRT GEGLU port; no placement changes or performance claims."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import types
import urllib.request

REFERENCE = 'results/flashrt-sm89/activation-gpu-placement-resume-2pe2mdrs/summary.json'
DIAGNOSTIC = 'results/flashrt-sm89/activation-placement-command-diyedfbl/diagnose-sm89-activation-placement.py'
DIAGNOSTIC_SHA = '8a6a0f8d686372ac4cdd2170e7d478c2a10f8a108e6e0b4c53783b38740ff263'
RUNTIME_KEYS = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG',
                'NVIDIA_TF32_OVERRIDE','OMP_PROC_BIND','OMP_PLACES','KMP_AFFINITY')
KEYS = ('action', 'action_final_raw', 'action_final')
SEQUENCE = ['fixed'] * 3 + ['changed-text'] * 3 + ['changed-state'] * 3 + ['fixed']


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def flat(response, key):
    value = response[key]
    need(isinstance(value, list) and len(value) == 10 and
         all(isinstance(row, list) and len(row) == 32 for row in value), key + ': shape must be 10x32')
    result = [v for row in value for v in row]
    need(all(type(v) in (int, float) and math.isfinite(v) for v in result), key + ': nonfinite/invalid action')
    return result


def compare(x, y):
    result = {}
    for key in KEYS:
        xx, yy = flat(x, key), flat(y, key)
        delta = [a-b for a, b in zip(xx, yy)]
        sq, norm = math.fsum(v*v for v in delta), math.fsum(v*v for v in yy)
        maximum = max(abs(v) for v in delta)
        result[key] = dict(max_abs=maximum, rmse=math.sqrt(sq/len(delta)),
                           relative_l2=math.sqrt(sq/norm) if norm else (0.0 if sq == 0 else None), exact=maximum == 0)
    return result


def exact(diff):
    return all(v['exact'] for v in diff.values())


def run(command, log, env=None, timeout=1800):
    command = [str(v) for v in command]
    print('运行:', ' '.join(command), flush=True)
    started = time.monotonic()
    with Path(log).open('w') as out:
        child = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while True:
                try:
                    code = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    need(elapsed < timeout, f'命令超时；日志 {log}')
                    print(f'处理中：{elapsed:.0f}s；日志 {Path(log).name}', flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=10)
    text = Path(log).read_text(errors='replace')
    need(code == 0, f'命令失败 ({code})，日志 {log}\n{text[-3000:]}')
    return text


def trace_records(text):
    return [dict(re.findall(r'(\w+)=([^\s]+)', line.split('flashrt_geglu:', 1)[1]))
            for line in text.splitlines() if 'flashrt_geglu:' in line]


def environment(saved, runtime, gpu, mode, graphs, trace=True):
    prefixes = ('GGML_', 'CUDA_', 'PI0_', 'PI05_', 'PI_MODEL', 'LLAMA_')
    env = {k:v for k,v in os.environ.items() if not k.startswith(prefixes)}
    for key in RUNTIME_KEYS:
        value = runtime[key]
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env.update(saved)
    for key in list(env):
        if key.startswith('LLAMA_ARG_') or key.startswith('PI0_') or key.startswith('PI05_') or 'SHADOW' in key and key.startswith('GGML_'):
            env.pop(key)
    env.pop('GGML_SCHED_DEBUG', None)
    env.pop('GGML_CUDA_DISABLE_GRAPHS', None)
    if not graphs:
        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    env.update(CUDA_VISIBLE_DEVICES=gpu, GGML_CUDA_FLASHRT='1', GGML_CUDA_FLASHRT_GEGLU=str(mode),
               GGML_CUDA_FLASHRT_BIAS_ACTIVATION='0', GGML_CUDA_FLASHRT_TRACE='1' if trace else '0')
    return env


def fixture_checks(prefix, text, mode, graphs):
    data = json.loads(Path(str(prefix)+'.json').read_text())
    raw = Path(str(prefix)+'.bin').read_bytes()
    records = data['records']
    need(data.get('status') == 'tests_complete' and data.get('input_update_checked') is True and
         len(records) == 24 and data['mode'] == mode and data['graphs_requested'] == graphs,
         'GEGLU fixture输出数量/运行设置不符')
    end = 0
    names = set()
    for item in records:
        offset, size = item['raw_offset_bytes'], item['raw_bytes']
        need(type(offset) is int and type(size) is int and offset == end and size > 0 and
             offset + size <= len(raw), 'fixture二进制范围错误')
        need(item['repeat_bit_mismatches'] == 0, 'fixture重放输出变化')
        need(item['finite'] is True and item['reference_bound_failures'] == 0, 'fixture数学参考未通过')
        need(item['dtype'] in ('f32','f16') and item['phase'] in ('initial','changed'), 'fixture类型/阶段错误')
        names.add((item['name'],item['phase']))
        end += size
    need(end == len(raw) and len(names) == 24, 'fixture二进制长度/唯一记录不符')
    hits = trace_records(text)
    if mode == 0:
        need(not hits, 'baseline意外命中新kernel')
    else:
        for dtype in ('f32','f16'):
            need(any(r.get('dtype') == dtype and r.get('width') == '4096' for r in hits), f'未命中{dtype} AE GEGLU')
            need(any(r.get('dtype') == dtype and r.get('width') == '16384' for r in hits), f'未命中{dtype} encoder GEGLU')
        need(not any(r.get('width') == '1024' for r in hits), '不支持width未回退')
        for item in records:
            if not item['eligible']:
                continue
            matching = [r for r in hits if r.get('node') == item['node']]
            need(matching, f'支持的fixture未命中：{item["name"]}')
            if graphs:
                need(any(r.get('capture') == '1' for r in matching), f'fixture未实际capture：{item["name"]}')
        if not graphs:
            need(all(r.get('capture') == '0' for r in hits), 'graphs-off发生capture')
    return data, raw, hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.home()/'Jetson-PI-Edge-flashrt-sm89')
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--gpu')
    args = ap.parse_args(argv)
    root = args.root.resolve(strict=True)
    out = Path(tempfile.mkdtemp(prefix='geglu-port-', dir=root/'results/flashrt-sm89'))
    report = dict(status='running', full_port_complete=False, stages={}, cleanup_errors=[],
                  performance='not_measured', kernel_math='FlashRT expf GEGLU; native F16 and GGML F32 dtype extension',
                  numerical_policy='Kernel uses independent reference tolerances. Model drift is reported, never automatically accepted.',
                  runner_sha256=sha(__file__))
    print('本次结果目录:', out, flush=True)
    try:
        refpath = (args.reference or root/REFERENCE).resolve(strict=True)
        ref = json.loads(refpath.read_text())
        need(ref.get('status') in ('placement_numeric_review_required','placement_and_fusion_exact_for_three_cases') and
             ref.get('cleanup_errors') == [], '缺少已完成的H/P/F对照记录')
        paths = {'runner_sha256':root/'scripts/run-sm89-activation-stage.py',
                 'helper_sha256':root/'run-ggml-math-shadow.py',
                 'micro_helper_sha256':root/'scripts/run-sm89-gemm-stage.py'}
        need(all(sha(path) == ref['helper_code_sha256'][k] for k,path in paths.items()), '既有helper源码变化；未导入')
        diagpath = root/DIAGNOSTIC
        need(sha(diagpath) == DIAGNOSTIC_SHA, 'GPU选择helper版本不符')
        a,h,micro = [types.SimpleNamespace(**runpy.run_path(str(p))) for p in paths.values()]
        d = types.SimpleNamespace(**runpy.run_path(str(diagpath)))
        need(h.listener_pid() is None, '18081已占用；未停止已有进程')
        cases = {c['id']:c for c in ref['input']['cases']}
        need(set(cases) == set(SEQUENCE), '参考案例不完整')
        targets = {}
        for graphs in (False,True):
            stage = ref['stages'][f'graphs-{"on" if graphs else "off"}-H']
            need(stage['status'] == 'passed' and len(stage['records']) == 22, 'H基线不完整')
            target = {}
            for item in stage['records']:
                response, name = item['response'],item['case']
                need(exact(compare(response,target.get(name,response))), 'H基线重复不一致')
                target[name] = response
            need(set(target) == set(cases), 'H基线案例不完整')
            targets[(graphs,0)] = target
        saved = ref['stages']['graphs-off-H']['environment']
        need(saved.get('GGML_CUDA_FLASHRT_GEMM') == '2' and saved.get('GGML_CUDA_FLASHRT_BIAS_ACTIVATION') == '0',
             'H基线feature配置不符')
        runtime = ref['inherited_runtime_environment']
        need(set(runtime) == set(RUNTIME_KEYS), '缺少运行环境记录')
        build = Path(ref['build']['directory']).resolve(strict=True)
        binary = build/'bin/llama-server'
        command = ref['stages']['graphs-off-H']['command']
        need(command == ref['stages']['graphs-on-H']['command'] and '--override-tensor' not in command and
             Path(command[0]).resolve() == binary.resolve(), '参考命令不是原放置H组')
        models = [Path(item['path']) for item in ref['models']]
        image = Path(ref['input']['image'])
        print('校验固定模型与输入...', flush=True)
        need({str(p):sha(p) for p in models} == ref['model_sha256'] and sha(image) == ref['input']['image_sha256'],
             '模型或输入图像变化')
        report.update(reference=str(refpath), reference_sha256=sha(refpath), input=ref['input'], models=ref['models'],
                      model_sha256=ref['model_sha256'], helper_code_sha256=ref['helper_code_sha256'])
        cmake = shutil.which('cmake') or str(Path.home()/'.local/bin/cmake')
        run([cmake,'-S',root,'-B',build,'-DGGML_CUDA=ON','-DGGML_CUDA_FLASHRT=ON',
             '-DCMAKE_CUDA_ARCHITECTURES=89','-DLLAMA_BUILD_TESTS=ON'],out/'configure.log')
        run([cmake,'--build',build,'--target','llama-server','test-flashrt-geglu','-j','8'],out/'build.log')
        def artifact_hashes():
            result = a.artifact_hashes(build)
            result['bin/test-flashrt-geglu'] = sha(build/'bin/test-flashrt-geglu')
            return result
        manifest = dict(sources=a.source_hashes(root,micro),artifacts=artifact_hashes(),
                        cache_sha256=sha(build/'CMakeCache.txt'))
        save(out/'build-manifest.json',manifest)
        report['build_manifest'] = str(out/'build-manifest.json')
        print(run([build/'bin/test-flashrt-geglu','--host-only'],out/'host.log'),end='',flush=True)
        if args.gpu:
            selection = {'selected_uuid':args.gpu}
            snapshot = a.wait_for_idle_gpu(args.gpu,out/'idle',micro)
            selection['selected_uuid'] = snapshot['uuid']
        else:
            selection = d.select_gpu(ref['gpu_selection']['selected_uuid'],out,a,micro)
        gpu = selection['selected_uuid']; report['gpu_selection'] = selection
        def verify():
            need(a.source_hashes(root,micro) == manifest['sources'] and artifact_hashes() == manifest['artifacts'] and
                 sha(build/'CMakeCache.txt') == manifest['cache_sha256'], '运行期间源码或二进制变化')
            need(sha(refpath) == report['reference_sha256'] and sha(image) == ref['input']['image_sha256'], '参考或输入变化')
        fixtures = {}
        report['fixtures'] = {}
        for graphs in (False,True):
            for mode in (0,1):
                label = f'fixtures-graphs-{"on" if graphs else "off"}-mode-{mode}'
                verify(); a.wait_for_idle_gpu(gpu,out/(label+'-idle'),micro)
                prefix = out/label
                text = run([build/'bin/test-flashrt-geglu','--mode',mode,'--graphs','on' if graphs else 'off',
                            '--output-prefix',prefix],out/(label+'.log'),environment(saved,runtime,gpu,mode,graphs))
                data, raw, hits = fixture_checks(prefix,text,mode,graphs)
                fixtures[(graphs,mode)] = (data,raw)
                report['fixtures'][label] = dict(status='passed',result=str(prefix)+'.json',trace_records=len(hits))
                if graphs:
                    old_data, old_raw = fixtures[(False,mode)]
                    fields = ('name','phase','dtype','raw_offset_bytes','raw_bytes','eligible')
                    need([[r[k] for k in fields] for r in data['records']] == [[r[k] for k in fields] for r in old_data['records']]
                         and raw == old_raw, '同一GEGLU模式Graph开关输出不一致')
                print(label+': 数学参考/重复与输入更新通过',flush=True)
                save(out/'summary.json',report)
        report['fixture_correctness'] = 'reference_and_graph_replay_passed_f32_f16'
        for graphs in (False,True):
            for mode in (0,1):
                label = f'model-graphs-{"on" if graphs else "off"}-mode-{mode}'
                verify(); a.wait_for_idle_gpu(gpu,out/(label+'-idle'),micro)
                need(h.listener_pid() is None, '18081被占用；未停止已有进程')
                env = environment(saved,runtime,gpu,mode,graphs)
                stage = dict(status='running',records=[],command=command,
                             environment={k:v for k,v in env.items() if k.startswith(('GGML_','CUDA_','PI_MODEL'))},
                             inherited_runtime_environment={k:env.get(k) for k in RUNTIME_KEYS})
                report['stages'][label] = stage
                logpath = out/(label+'.log')
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                repeated = {}
                print('阶段:',label,flush=True)
                with logpath.open('w') as log:
                    child = subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    stage['owned_pid'] = child.pid
                    try:
                        h.wait_ready(child,opener)
                        for i,name in enumerate(SEQUENCE,1):
                            need(child.poll() is None and h.listener_pid() == child.pid,'服务身份变化')
                            h.verify_server_identity(child.pid,binary,*models)
                            need(h.request(opener,'POST','/foreground/reset').get('reset') is True,'reset失败')
                            need(h.request(opener,'POST','/foreground/image',{'path':str(image)},timeout=30).get('loaded') is True,'image失败')
                            need(h.request(opener,'PUT','/foreground/state',{'state':cases[name]['state']}).get('state_dim') == 32,'state失败')
                            response = h.request(opener,'POST','/foreground/infer',{'text':cases[name]['text']},timeout=120)
                            record = dict(case=name,response=a.safe_json(response))
                            stage['records'].append(record); save(out/'summary.json',report)
                            diff = compare(response,targets[(graphs,0)][name]); record['vs_ggml'] = diff
                            need(diff['action']['exact'],'初始噪声变化')
                            need(exact(compare(response,repeated.get(name,response))), '同输入重复输出变化')
                            if mode == 0:
                                need(exact(diff),'GEGLU关闭时已验收H基线变化')
                            repeated[name] = response
                            print(f'{label}: {i}/10 {name}, max_abs={diff["action_final_raw"]["max_abs"]:.9g}',flush=True)
                    except BaseException as error:
                        stage.update(status='failed', error=f'{type(error).__name__}: {error}')
                        raise
                    finally:
                        try:
                            h.stop_child(child)
                        except BaseException as error:
                            report['cleanup_errors'].append(f'{label} owned PID {child.pid}: {error}')
                        save(out/'summary.json',report)
                need(not report['cleanup_errors'],'自有服务清理未完成')
                hits = trace_records(logpath.read_text(errors='replace'))
                if mode == 0:
                    need(not hits,'模型baseline意外命中新GEGLU')
                else:
                    need(any(r.get('dtype') == 'f32' and r.get('width') == '4096' and r.get('rows') == '11' for r in hits),
                         '真实decoder GEGLU未命中')
                    need(any(r.get('dtype') == 'f32' and r.get('width') == '16384' for r in hits),'真实encoder GEGLU未命中')
                    if graphs:
                        need(any(r.get('capture') == '1' for r in hits),'允许Graph的模型未观察到GEGLU capture')
                    else:
                        need(all(r.get('capture') == '0' for r in hits),'graphs-off意外capture')
                stage.update(status='passed',trace_records=len(hits),capture_records=sum(r.get('capture')=='1' for r in hits))
                targets[(graphs,mode)] = repeated
                save(out/'summary.json',report)
        verify()
        report['model_comparisons'] = {name:compare(targets[(False,1)][name],targets[(False,0)][name]) for name in cases}
        report['model_graph_comparisons'] = {name:compare(targets[(True,1)][name],targets[(False,1)][name]) for name in cases}
        need(all(exact(v) for v in report['model_graph_comparisons'].values()), 'FlashRT GEGLU模型Graph开关不一致')
        same = all(exact(v) for v in report['model_comparisons'].values())
        report.update(status='geglu_kernel_integrated',
                      model_numerics='exact_for_three_cases' if same else 'drift_recorded_task_accuracy_not_accepted')
        print('\ncase,max_abs,RMSE,relative_L2',flush=True)
        for name,diff in report['model_comparisons'].items():
            v=diff['action_final_raw']; print(f'{name},{v["max_abs"]:.9g},{v["rmse"]:.9g},{v["relative_l2"]}',flush=True)
        return 0
    except BaseException as error:
        report.update(status='incomplete',error=f'{type(error).__name__}: {error}')
        for stage in report['stages'].values():
            if stage.get('status') == 'running':
                stage.update(status='failed', error=report['error'])
        print(report['error'],flush=True)
        return 130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        save(out/'summary.json',report)
        print('status:',report['status'],flush=True)
        if 'model_numerics' in report:
            print('model_numerics:',report['model_numerics'],flush=True)
        print('已保存:',out/'summary.json',flush=True)
        print('本次是GEGLU kernel移植；未宣称完整FlashRT移植或性能提升。',flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
