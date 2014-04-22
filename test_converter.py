import hashlib
import unittest
import numpy as np

from hyperopt import hp
import hpconvnet.pyll_slm
import convert_convnet_searchspace

class Converter_Test(unittest.TestCase):
    def test_convert_hyperopt_param(self):
        pass

    def test_switch_multiple_types(self):
        prefix = "test"
        order = hp.choice('%sp_order' % prefix,
                [1, 2, hp.loguniform('%sp_order_real' % prefix,
                    mu=np.log(1), sigma=np.log(3))])
        rval = convert_convnet_searchspace.convert_switch(order, 0)
        expected = '"testp_order": hp.choice("testp_order", [\n'\
                    '    {\n' \
                    '    "testp_order": 0,\n' \
                    '    },\n' \
                    '    {\n' \
                    '    "testp_order": 1,\n' \
                    '    },\n' \
                    '    {\n' \
                    '    "testp_order": 2,\n' \
                    '    "testp_order_real": hp.loguniform("testp_order_real", lower=0.0, upper=1.09861228867),\n' \
                    '    },\n' \
                    ']),\n'
        self.assertEqual(rval, expected)

    def test_loguniform(self):
        prefix = "test"
        # For whatever reason, this returns a float
        param = hp.loguniform('%sp_order_real' % prefix,
                    mu=np.log(0.1), sigma=np.log(10))
        expected = '"testp_order_real": hp.loguniform("testp_order_real", lower=-2.30258509299, upper=2.30258509299),\n'
        rval = convert_convnet_searchspace.convert_float(param, 0)
        self.assertEqual(rval, expected)