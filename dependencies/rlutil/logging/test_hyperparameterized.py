import six
import unittest

from rlutil.logging.hyperparameterized import Hyperparameterized, extract_hyperparams, CLSNAME, HyperparamWrapper


@six.add_metaclass(Hyperparameterized)
class Algo1(object):
    def __init__(self, hyper1=1.0, hyper2=2.0, model1=None):
        pass


class Algo2(Algo1):
    def __init__(self, hyper3=5.0, **kwargs):
        super(Algo2, self).__init__(**kwargs)


@six.add_metaclass(Hyperparameterized)
class Model1(object):
    def __init__(self, hyper1=None):
        pass

def get_params_json(**kwargs):
    hyper_dict = extract_hyperparams(HyperparamWrapper(**kwargs))
    del hyper_dict[CLSNAME]
    return hyper_dict

class HyperparameterizedTest(unittest.TestCase):
    def testHyperparams(self):
        m1 = Model1(hyper1='Test')
        a1 = Algo2(hyper1=1.0, hyper2=5.0, hyper3=10.0, model1=m1)

        self.assertIsInstance(type(a1), Hyperparameterized)
        params = get_params_json(a1=a1)['a1']
        expected_params = {'__clsname__': 'Algo2', 
                            'hyper1': 1.0, 
                            'hyper2': 5.0, 
                            'hyper3': 10.0, 
                            'model1': {'__clsname__': 'Model1', 'hyper1': 'Test'}}
        self.assertEqual(params, expected_params)

if __name__ == "__main__":
    unittest.main()