import json
import os
import sys
from multiprocessing import Process, Queue

from bremen.albrecht import evaluate

debug = False


def _child(queue, *args, **kwargs):
    result = evaluate(*args, **kwargs)
    queue.put(result)


def eval_(*args, **kwargs):
    if debug:
        return evaluate(*args, **kwargs)
    # Pass the queue explicitly: macOS uses spawn, so globals are not shared.
    queue = Queue()
    process = Process(target=_child, args=(queue,) + args, kwargs=kwargs)
    process.start()
    process.join()
    if process.exitcode:
        queue.close()
        raise RuntimeError(f'CoDe-DTI evaluation child exited with {process.exitcode}')
    try:
        return queue.get(timeout=5)
    finally:
        queue.close()


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('configuration_json_file', type=open)
    p.add_argument('--output-dir', '-o', default=".", type=str)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    global debug
    debug = args.debug

    configuration = json.load(args.configuration_json_file)
    cv_mode = configuration['cv_mode']
    parameters = configuration['parameters']
    n_fold = configuration['n_fold']
    if configuration['method'] == 'gp':
        gp_search(cv_mode,
                  configuration['intmat'], configuration['fingerprint'], configuration['fingerprint_bit'],
                  parameters, n_fold, configuration.get('max_iter'), args.output_dir)
    else:
        random_search(cv_mode, configuration['intmat'], configuration['fingerprint'], configuration['fingerprint_bit'],
                      parameters, n_fold, configuration['max_iter'], args.output_dir)
    return 0


def gp_search(cv_mode, intmat, fingerprint, fingerprint_bit, parameters, n_fold, max_iter, output_dir):
    raise NotImplementedError('The historical repository does not include parameter_search.bayes; use method=random')


def random_search(cv_mode, intmat, fingerprint, fingerprint_bit, parameters, n_fold, max_iter, output_dir):
    import random
    results = []
    for i in range(max_iter):
        param = {
            key: random.choice(value) for key, value in parameters.items()
        }
        output_dir_ = os.path.join(output_dir, 'iter_%d' % i)
        os.makedirs(output_dir_, 0o755, exist_ok=True)
        with open(os.path.join(output_dir_, 'param.json'), 'w')as fp:
            json.dump(param, fp, indent=2)
        result = eval_(cv_mode=cv_mode, intmat_path=intmat,
                       fingerprint_bit=fingerprint_bit, fingerprint_path=fingerprint,
                       n_fold=n_fold, output_dir=output_dir_, **param)
        print('param:', param, 'result:', result)
        param['result'] = result
        results.append(param)

        with open(os.path.join(output_dir, 'result_dump.json'), 'w') as fp:
            json.dump(results, fp)


if __name__ == '__main__':
    sys.exit(main())
