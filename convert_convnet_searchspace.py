import hpconvnet
import hpconvnet.cifar10
import hyperopt

import cStringIO

search_space = hpconvnet.cifar10.build_search_space(max_n_features=4500,
                                                    bagging_fraction=0.5,
                                                    n_unsup=2000,
                                                    abort_on_rows_larger_than=50 * 1000)

pipeline = search_space["pipeline"]
variables = dict()
nodes = dict()


def convert_hyperopt_param(arg, indent):
    label = arg.inputs()[0]._obj
    name = arg.inputs()[1].name
    rval = globals()["convert_" + name](arg.inputs()[1], indent, label)
    variables[label] = rval
    return rval


def convert_literal(arg, indent):
    if type(arg._obj) == hyperopt.pyll.base.SymbolTableEntry:
        #This function has no effect on the search space
        if arg._obj.apply_name == "slm_img_uint8_to_float32":
            pass
        else:
            #print " "*indent + arg.name + " " + str(arg._obj)
            pass
    else:
        #return " "*indent + str(arg._obj) + ",\n"
        return str(arg._obj)


def convert_container(arg, indent, child_indent=0):
    recursion_rvals = dict()
    for dict_descendant in arg.inputs():
        if dict_descendant.name == "literal":
            continue
            # print " "*indent + dict_descendent.name
        rval = searchspace_to_string(dict_descendant, indent + child_indent)
        if rval is not None:
            recursion_rvals.update(rval)
        #fh.write(" "*indent + "},\n")
    #return "".join(recursion_rvals)
    return recursion_rvals


def convert_partial(arg, indent):
    return convert_container(arg, indent)


def convert_pos_args(arg, indent):
    return convert_container(arg, indent)


def convert_dict(arg, indent):
    return convert_container(arg, indent, 4)


def convert_mul(arg, indent):
    return convert_container(arg, indent, 4)


def convert_slm_uniform_M_FB(arg, indent):
    return convert_container(arg, indent, 4)


def convert_getitem(arg, indent):
    return convert_container(arg, indent, 4)


def convert_getattr(arg, indent):
    return convert_container(arg, indent, 4)


def convert_div(arg, indent):
    return convert_container(arg, indent)


def convert_switch(arg, indent):
    """
    This function will screw up hp.pchoice because it removes duplicate return
    values.
    """
    # First input is a hyperopt_param randint
    #dict_rval = cStringIO.StringIO()
    recursion_rvals = dict()
    assert arg.inputs()[0].name == "hyperopt_param", arg.inputs()[0]
    randint_node = arg.pos_args[0]
    label = randint_node.inputs()[0]._obj
    #dict_rval.write(" " * indent + "\"" + label + "\"" + ": hp.choice(\"" +
    #           label + "\", [\n")

    # If a descendants is a literal, replace it with its indices
    do_recursion = [descendant.name != "literal" for descendant in arg.inputs()[1:]]
    children = []
    for i, switch_descendant in enumerate(arg.inputs()[1:]):
        children.append(dict())
        children[-1][label] = i
        do_print = False
        tmp = None
        if do_recursion[i]:
            tmp = searchspace_to_string(switch_descendant, indent + 4)
            #print label, tmp
            children[-1].update(tmp)
        #dict_rval.write(" " * (indent + 4) + "},\n")

    #dict_rval.write(" " * indent + "]),\n")
    #dict_rval = dict_rval.getvalue()
    variables[label] = label
    nodes[label] = '%s = hp.choice("%s", [%s])' % (str(label), str(label),
                                                 ", ".join(["{" + ", ".join(['"%s": %s' % (str(c), str(child[c])) for c in child]) + "}" for child in children]))
    #return dict_rval
    return {label: label}


def convert_float(arg, indent):
    assert len(arg.inputs()) == 1, arg.inputs()
    name = arg.inputs()[0].name
    if name == "literal":
        pass
    elif name == "hyperopt_param":
        return convert_hyperopt_param(arg.inputs()[0], indent)
    else:
        return searchspace_to_string(arg.inputs()[0], indent)


def convert_loguniform(arg, indent, label):
    assert len(arg.inputs()) == 2
    assert all([input.name == "literal" for input in arg.inputs()]), arg.inputs()
    lower = arg.inputs()[0]._obj
    upper = arg.inputs()[1]._obj
    dict_node = '"%s": hp.loguniform("%s", lower=%s, upper=%s),\n' % (label, label, lower, upper)
    var_node = '%s = hp.loguniform("%s", %s, %s)' % (label, label, lower, upper)
    nodes[label] = var_node
    #return ' '*indent + dict_node
    return {label: label}


def convert_uniform(arg, indent, label):
    assert len(arg.inputs()) == 2
    assert all([input.name == "literal" for input in arg.inputs()]), arg.inputs()
    lower = arg.inputs()[0]._obj
    upper = arg.inputs()[1]._obj
    dict_node = '"%s": hp.uniform("%s", lower=%s, upper=%s),\n' % (label, label, lower, upper)
    var_node = '%s = hp.uniform("%s", %s, %s)' % (label, label, lower, upper)
    nodes[label] = var_node
    #return ' '*indent + dict_node
    return {label: label}


def convert_quniform(arg, indent, label):
    assert len(arg.inputs()) == 3
    assert all([input.name == "literal" for input in arg.inputs()]), arg.inputs()
    lower = arg.inputs()[0]._obj
    upper = arg.inputs()[1]._obj
    q = arg.inputs()[2]._obj
    dict_node = '"%s": hp.quniform("%s", lower=%s, upper=%s, q=%s),\n' % (label, label, lower, upper, q)
    var_node = '%s = hp.quniform("%s", %s, %s, %s)' % (label, label, lower, upper, q)
    nodes[label] = var_node
    #return ' '*indent + dict_node
    return {label: label}


def convert_normal(arg, indent, label):
    assert len(arg.inputs()) == 2
    assert all([input.name == "literal" for input in arg.inputs()]), arg.inputs()
    mu = arg.inputs()[0]._obj
    sigma = arg.inputs()[1]._obj
    dict_node = '"%s": hp.normal("%s", mu=%s, sigma=%s),\n' % (label, label, mu, sigma)
    var_node = '%s = hp.normal("%s", %s, %s)' % (label, label, mu, sigma)
    nodes[label] = var_node
    #return ' '*indent + dict_node
    return {label: label}


def convert_lognormal(arg, indent, label):
    assert len(arg.inputs()) == 2
    assert all([input.name == "literal" for input in arg.inputs()]), arg.inputs()
    mu = arg.inputs()[0]._obj
    sigma = arg.inputs()[1]._obj
    dict_node = '"%s": hp.lognormal("%s", mu=%s, sigma=%s),\n' % (label, label, mu, sigma)
    var_node = '%s = hp.lognormal("%s", %s, %s)' % (label, label, mu, sigma)
    nodes[label] = var_node
    #return ' '*indent + dict_node
    return {label: label}


def convert_qloguniform(arg, indent, label):
    assert len(arg.inputs()) == 3
    assert all([input.name == "literal" for input in arg.inputs()]), arg.inputs()
    lower = arg.inputs()[0]._obj
    upper = arg.inputs()[1]._obj
    q = arg.inputs()[2]._obj
    dict_node = '"%s": hp.qloguniform("%s", lower=%s, upper=%s, q=%s),\n' % (label, label, lower, upper, q)
    var_node = '%s = hp.qloguniform("%s", %s, %s, %s)' % (label, label, lower, upper, q)
    nodes[label] = var_node
    #return ' '*indent + dict_node
    return {label: label}


def convert_qnormal(arg, indent, label):
    assert len(arg.inputs()) == 3
    assert all([input.name == "literal" for input in arg.inputs()]), arg.inputs()
    mu = arg.inputs()[0]._obj
    sigma = arg.inputs()[1]._obj
    q = arg.inputs()[2]._obj
    dict_node = '"%s": hp.qnormal("%s", mu=%s, sigma=%s, q=%s),\n' % (label, label, mu, sigma, q)
    var_node = '%s = hp.qnormal("%s", %s, %s, %s)' % (label, label, mu, sigma, q)
    nodes[label] = var_node
    #return ' '*indent + dict_node
    return {label: label}


def convert_qlognormal(arg, indent, label):
    assert len(arg.inputs()) == 3
    assert all([input.name == "literal" for input in arg.inputs()]), arg.inputs()
    mu = arg.inputs()[0]._obj
    sigma = arg.inputs()[1]._obj
    q = arg.inputs()[2]._obj
    dict_node= '"%s": hp.qlognormal("%s", mu=%s, sigma=%s),\n' % (label, label, mu, sigma, q)
    var_node = '%s = hp.qlognormal("%s", %s, %s)' % (label, label, mu, sigma, q)
    nodes[label] = var_node
    #return ' '*indent + dict_node
    return {label: label}


def convert_categorical(arg, label):
    raise NotImplementedError()
    # No difference to a randint node
    upper = arg.inputs()[1]._obj
    #print '%s {%s} [0]' % \
    #      (label, ", ".join(["%s" % (i,) for i in range(upper)]))


def convert_randint(arg, label):
    raise NotImplementedError()
    assert len(arg.inputs()) == 1
    #randint['x', 5] -> x [0, 4]i [0]
    upper = arg.inputs()[0]._obj
    #print '%s {%s} [0]' % \
    #      (label, ", ".join(["%s" % (i,) for i in range(upper)]))


def searchspace_to_string(expr, indent=0):
    arg = expr
    # print " "*indent + arg.name
    label = "TODO"

    if arg.name == "hyperopt_param":
        convert_hyperopt_param(arg, indent)
    elif arg.name == "literal":
        convert_literal(arg, indent)
    elif arg.name == "dict":
        return convert_dict(arg, indent)
    elif arg.name == "switch":
        return convert_switch(arg, indent)
    elif arg.name == "float":
        return convert_float(arg, indent)
    elif arg.name == "pos_args":
        return convert_pos_args(arg, indent)
    elif arg.name == "partial":
        return convert_partial(arg, indent)
    elif arg.name == "div" or arg.name == "mul":
        return convert_div(arg, indent)
    elif arg.name == "slm_uniform_M_FB":
        return convert_slm_uniform_M_FB(arg, indent)
    elif arg.name == "getitem":
        return convert_getitem(arg, indent)
    elif arg.name == "getattr":
        return convert_getattr(arg, indent)
    elif arg.name == "pyll_theano_batched_lmap":
        return convert_dict(arg, indent)
    elif arg.name == "cifar10_unsup_images":
        return convert_dict(arg, indent)
    elif arg.name == "floordiv":
        return convert_dict(arg, indent)
    elif arg.name == "pow":
        return convert_dict(arg, indent)
    elif arg.name == "name":
        raise NotImplementedError()
    elif arg.name == "int":
        return convert_float(arg, indent)
    elif arg.name == "fb_whitened_projections":
        return convert_dict(arg, indent)
    elif arg.name == "random_patches":
        return convert_dict(arg, indent)
    elif arg.name == "np_RandomState":
        return convert_dict(arg, indent)
    elif arg.name == "patch_whitening_filterbank_X":
        return convert_dict(arg, indent)
    elif arg.name == "fb_whitened_patches":
        return convert_dict(arg, indent)
    elif arg.name == "ceildiv":
        return convert_dict(arg, indent)
    elif arg.name == "max":
        return convert_dict(arg, indent)
    elif arg.name == "add":
        return convert_dict(arg, indent)
    elif arg.name == "sub":
        return convert_dict(arg, indent)
    elif arg.name == "randint":
        return convert_randint(arg, label)
    elif arg.name == "categorical":
        return convert_categorical(arg, label)
    elif arg.name == "uniform":
        return convert_uniform(arg, indent, label)
    elif arg.name == "quniform":
        return convert_quniform(arg, indent, label)
    elif arg.name == "loguniform":
        return convert_loguniform(arg, label)
    elif arg.name == "qloguniform":
        return convert_qloguniform(arg, label)
    elif arg.name == "normal":
        return convert_normal(arg, indent, label)
    elif arg.name == "qnormal":
        return convert_qnormal(arg, label)
    elif arg.name == "lognormal":
        return convert_lognormal(arg, label)
    elif arg.name == "qlognormal":
        return convert_qlognormal(arg, label)
    else:
        raise Exception("Node name %s not known" % arg.name)
    # pprint(arg, fh, lineno, indent + 2)

pipeline = hyperopt.pyll.as_apply(pipeline)

if __name__ == "__main__":
    print "WARNING: Make sure you don't have hp.pchoice objects in your" \
          " expression graph because they will not be translated correctly."
    file = searchspace_to_string(pipeline)
    assert len(variables) == 238, "Got an unexpected amount of variables (%d)" % len(variables)
    fh = open("output.py", "w")
    fh.write("from hyperopt import hp\n")
    keys = []
    for key in nodes:
        keys.append(key)
    keys.sort()
    keys.reverse()
    for key in keys:
        print key, nodes[key]
        fh.write(nodes[key] + "\n")
    fh.write("space = " + file.keys()[0])
    fh.close()