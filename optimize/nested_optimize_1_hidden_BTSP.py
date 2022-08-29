from EIANN import *
from EIANN_utils import *
from nested.utils import Context, param_array_to_dict


context = Context()


def config_worker():
    context.start_instance = int(context.start_instance)
    context.num_instances = int(context.num_instances)
    context.network_id = int(context.network_id)
    context.task_id = int(context.task_id)


def get_random_seeds():
    return [[int.from_bytes((context.network_id, context.task_id, instance_id), byteorder='big')
            for instance_id in
             range(context.start_instance, context.start_instance + context.num_instances)]]


def compute_features(x, seed, model_id=None, export=False):
    """

    :param x: array of float
    :param model_id: str
    :param export: bool
    :return: dict
    """
    print(seed)
    param_dict = param_array_to_dict(x, context.param_names)

    layer_config = {'Input':
                        {'E':
                             {'size': 21}
                         },
                    'H1':
                        {'E':
                             {'size': 7,
                              'activation': 'relu'
                              },
                         'FBI':
                             {'size': 1,
                              'activation': 'relu'
                              },
                         'Dend_I':
                             {'size': 1,
                              'activation': 'relu'
                              }
                         },
                    'Output':
                        {'E':
                             {'size': 21,
                              'activation': 'relu'
                              },
                         'FBI':
                             {'size': 1,
                              'activation': 'relu'
                              }
                         }
                    }

    H1_E_Input_E_max_weight = param_dict['H1_E_Input_E_max_weight']
    H1_E_Input_E_max_init_weight = H1_E_Input_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                   layer_config['Input']['E']['size']
    H1_E_BTSP_learning_rate = param_dict['H1_E_BTSP_learning_rate']
    H1_E_BTSP_pos_loss_th = param_dict['H1_E_BTSP_pos_loss_th']
    H1_E_BTSP_neg_loss_th = param_dict['H1_E_BTSP_neg_loss_th']
    H1_E_Output_E_max_weight = param_dict['H1_E_Output_E_max_weight']
    H1_E_Output_E_min_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_min_init_weight_factor'] / \
                                    layer_config['Output']['E']['size']
    H1_E_Output_E_max_init_weight = H1_E_Output_E_max_weight * param_dict['FB_BTSP_max_init_weight_factor'] / \
                                    layer_config['Output']['E']['size']

    H1_E_H1_FBI_weight = param_dict['H1_E_H1_FBI_weight']
    H1_I_dend_H1_E_weight = param_dict['H1_I_dend_H1_E_weight']
    H1_E_H1_I_dend_init_weight = param_dict['H1_E_H1_I_dend_init_weight']
    H1_E_H1_I_dend_learning_rate = param_dict['H1_E_H1_I_dend_learning_rate']

    Output_E_H1_E_max_weight = param_dict['Output_E_H1_E_max_weight']
    Output_E_H1_E_max_init_weight = Output_E_H1_E_max_weight * param_dict['FF_BTSP_max_init_weight_factor'] / \
                                    layer_config['H1']['E']['size']
    Output_E_BTSP_learning_rate = param_dict['Output_E_BTSP_learning_rate']
    Output_E_BTSP_pos_loss_th = param_dict['Output_E_BTSP_pos_loss_th']
    Output_E_BTSP_neg_loss_th = param_dict['Output_E_BTSP_neg_loss_th']
    Output_E_Output_FBI_weight = param_dict['Output_E_Output_FBI_weight']

    FBI_E_weight = param_dict['FBI_E_weight']

    projection_config = {'H1':
                             {'E':
                                  {'Input':
                                       {'E':
                                            {'weight_init': 'uniform_',
                                             'weight_init_args': (0, H1_E_Input_E_max_init_weight),
                                             'weight_bounds': (0, H1_E_Input_E_max_weight),
                                             'direction': 'F',
                                             'learning_rule': 'BTSP',
                                             'learning_rule_kwargs':
                                                 {'pos_loss_th': H1_E_BTSP_pos_loss_th,
                                                  'neg_loss_th': H1_E_BTSP_neg_loss_th,
                                                  'learning_rate': H1_E_BTSP_learning_rate
                                                  }
                                             }
                                        },
                                   'H1':
                                       {'FBI':
                                            {'weight_init': 'fill_',
                                             'weight_init_args': (H1_E_H1_FBI_weight,),
                                             'direction': 'R',
                                             'learning_rule': None
                                             },
                                        'Dend_I':
                                            {'weight_init': 'fill_',
                                             'weight_init_args': (H1_E_H1_I_dend_init_weight,),
                                             'weight_bounds': (None, 0),
                                             'direction': 'B',
                                             'compartment': 'dend',
                                             'learning_rule': 'DendriticLoss',
                                             'learning_rule_kwargs':
                                                 {'sign': -1,
                                                  'learning_rate': H1_E_H1_I_dend_learning_rate
                                                  }
                                             }
                                        },
                                   'Output':
                                       {'E':
                                            {'weight_init': 'uniform_',
                                             'weight_init_args': (H1_E_Output_E_min_init_weight,
                                                                  H1_E_Output_E_max_init_weight),
                                             'weight_bounds': (0, H1_E_Output_E_max_weight),
                                             'direction': 'B',
                                             'compartment': 'dend',
                                             'learning_rule': 'BTSP'
                                             }
                                        }
                                   },
                              'FBI':
                                  {'H1':
                                       {'E':
                                            {'weight_init': 'fill_',
                                             'weight_init_args': (FBI_E_weight,),
                                             'direction': 'F',
                                             'learning_rule': None
                                             }
                                        }
                                   },
                              'Dend_I':
                                  {'H1':
                                       {'E':
                                            {'weight_init': 'fill_',
                                             'weight_init_args': (H1_I_dend_H1_E_weight,),
                                             'direction': 'B',
                                             'learning_rule': None
                                             }
                                        }
                                   }
                              },
                         'Output':
                             {'E':
                                  {'H1':
                                       {'E':
                                            {'weight_init': 'uniform_',
                                             'weight_init_args': (0, Output_E_H1_E_max_init_weight),
                                             'weight_bounds': (0, Output_E_H1_E_max_weight),
                                             'direction': 'F',
                                             'learning_rule': 'BTSP',
                                             'learning_rule_kwargs':
                                                 {'pos_loss_th': Output_E_BTSP_pos_loss_th,
                                                  'neg_loss_th': Output_E_BTSP_neg_loss_th,
                                                  'learning_rate': Output_E_BTSP_learning_rate
                                                  }
                                             }
                                        },
                                   'Output':
                                       {'FBI':
                                            {'weight_init': 'fill_',
                                             'weight_init_args': (Output_E_Output_FBI_weight,),
                                             'direction': 'R',
                                             'learning_rule': None
                                             }
                                        }
                                   },
                              'FBI':
                                  {'Output':
                                       {'E':
                                            {'weight_init': 'fill_',
                                             'weight_init_args': (FBI_E_weight,),
                                             'direction': 'F',
                                             'learning_rule': None
                                             }
                                        }
                                   }
                              }
                         }

    hyperparameter_kwargs = {'tau': 3,
                             'forward_steps': 10,
                             'backward_steps': 0,
                             'learning_rate': 9.553728E-01,
                             'seed': seed
                             }

    input_size = 21
    dataset = torch.eye(input_size)
    target = torch.eye(dataset.shape[0])

    epochs = context.epochs

    network = EIANN(layer_config, projection_config, **hyperparameter_kwargs)

    if context.plot:
        for sample in dataset:
            network.forward(sample, store_history=True)
        plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=context.supervised, label='Initial')
        network.reset_history()

    network.train(dataset, target, epochs, store_history=True, shuffle=True, status_bar=context.status_bar)

    loss_history, epoch_argmax_accuracy = \
        analyze_EIANN_loss(network, target, supervised=context.supervised, plot=context.plot)

    final_epoch_loss = torch.mean(loss_history[-target.shape[0]:])
    final_argmax_accuracy = torch.mean(epoch_argmax_accuracy[-context.num_epochs_argmax_accuracy:])

    if context.plot:
        plot_EIANN_activity(network, num_samples=dataset.shape[0], supervised=context.supervised, label='Final')

    return {'loss': final_epoch_loss,
            'accuracy': final_argmax_accuracy}


def filter_features(primitives, current_features, model_id=None, export=False):

    features = {}
    for instance_features in primitives:
        for key, val in instance_features.items():
            if key not in features:
                features[key] = []
            features[key].append(val)
    for key, val in features.items():
        features[key] = np.mean(val)

    return features


def get_objectives(features, model_id=None, export=False):
    objectives = features
    return features, objectives
