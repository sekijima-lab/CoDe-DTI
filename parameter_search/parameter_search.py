import json
import os
import sys
from multiprocessing import Process, Queue

from bremen.albrecht import evaluate
from parameter_search.bayes import BayesianOptimizer

q = Queue()
debug = False


def _child(*args, **kwargs):
    result = evaluate(*args, **kwargs)
    q.put(result)


def eval_(*args, **kwargs):
    p = Process(target=_child, args=args, kwargs=kwargs)
    if debug:
        p.run()
    else:
        p.start()
    p.join()
    return q.get(timeout=5)


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
    bo = BayesianOptimizer(parameters)
    results = []
    for i, param in enumerate(bo.supply_next_param(max_iter)):
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
        bo.report(result['avg_roc_auc'])

        with open(os.path.join(output_dir, 'result_dump.json'), 'w') as fp:
            json.dump(results, fp)


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
