import unittest
import EIANN as ei
import EIANN.utils as eiu

class TestEIANN_0hidden(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize a 1 layer network to verify that it creates without error

        # Import test config from yaml

        layer_config = {'Input':
                            {'E':
                                 {'size': 7}},
                        'Output':
                            {'E':
                                 {'size': 21,
                                  'activation': 'softplus',
                                  'activation_kwargs':
                                      {'beta': 4.}
                                  },
                             'FBI':
                                 {'size': 1,
                                  'activation': 'softplus',
                                  'activation_kwargs':
                                      {'beta': 4.}
                                  }
                             }
                        }

        projection_config = {'Output':
                                 {'E':
                                      {'Input':
                                           {'E':
                                                {'weight_init': 'uniform_',
                                                 'weight_init_args': (0, 1),
                                                 'weight_bounds': (0, None),
                                                 'direction': 'FF',
                                                 'learning_rule': 'Backprop'
                                                 }
                                            },
                                       'Output':
                                           {'FBI':
                                                {'weight_init': 'fill_',
                                                 'weight_init_args': (-3.838023E+00,),
                                                 'direction': 'FB',
                                                 'learning_rule': None
                                                 }
                                            }
                                       },
                                  'FBI':
                                      {'Output':
                                           {'E':
                                                {'weight_init': 'fill_',
                                                 'weight_init_args': (1,),
                                                 'direction': 'FF',
                                                 'learning_rule': None
                                                 }
                                            }
                                       }
                                  }
                             }

        hyperparameter_kwargs = {'tau': 3,
                                 'forward_steps': 10,
                                 'backward_steps': 10,
                                 'learning_rate': 2.999993E+00,
                                 'seed': 42
                                 }

        self.network = ei.EIANN(layer_config, projection_config, **hyperparameter_kwargs)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.network.layers['Output'].populations['E'].projections['Input']['E'].weight.shape == (21,7)
        self.network.layers['Output'].populations['E'].projections['Input']['E'].weight.requires_grad == True
        self.network.layers['Output'].populations['E'].projections['Input']['E'].bias == None

    def test_plotting(self):
        eiu.test_EIANN_config(self.network,
                          dataset=n_hot_patterns(n=2, length=7),
                          target=torch.eye(dataset.shape[0]),
                          epochs=300)

    def test_learning:
        # check that final backprop loss is smaller initial loss

# def run_test(self):
#     try:
#         myFunc()
#     except ExceptionType:
#         self.fail("myFunc() raised ExceptionType unexpectedly!")
#


if __name__ == '__main__':
    unittest.main()
