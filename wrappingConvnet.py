import ast
import logging
# import sys
import time

# from shovel import task

import hyperopt
import hpconvnet.cifar10
import hpconvnet.slm

import HPOlib.benchmark_util as benchmark_util

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib-hpconvnet")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib-hpconvnet.wrappingConvnet")


def wrapping_convnet(params, **kwargs):
    print "WARNING: make sure that the number of startup jobs for tpe is set to"\
          " 50 and that kwargs are ignored."
    print params
    # Don't forget to set the number of startup jobs to 50...
    space = hpconvnet.cifar10.build_search_space(
        max_n_features=4000,
        bagging_fraction=1.0,
        n_unsup=7500, 
        abort_on_rows_larger_than=500 * 1000,  # -- elements
        output_sizes=(32, 64),
        # This is not part of the original search space and was 2 secs
        batched_lmap_speed_thresh = {'seconds': 100.0, 'elements': 150}
        )

    pipe = hyperopt.pyll.stochastic.recursive_set_rng_kwarg(space['pipeline'])
    hps = {}
    hyperopt.pyll_utils.expr_to_config(pipe, (), hps)
    memo = {}

    for param in params:
        node = hps[param]['node']
        print "###"
        print "Node:", node
        try:
            value = ast.literal_eval(str(params[param]))
        except ValueError as e:
            print "Malformed String:", params[param]
            raise e
        memo[node] = hyperopt.pyll.Literal(value)
        print "Memo[node]:", memo[node]
        print "Value", value
    space = hyperopt.pyll.as_apply(space)
    ctrl = hyperopt.Ctrl(None, current_trial={})
    print ctrl, vars(ctrl)
    rval = hpconvnet.cifar10.uslm_eval(space, memo, ctrl)
    print rval
    return rval


def main(params, **kwargs):
    print 'Params: ', params, '\n'
    y =  wrapping_convnet(params, **kwargs)
    print 'Result: ', y
    return y


if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))

