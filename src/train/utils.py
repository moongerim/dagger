import os
import pprint
import scipy.io
# from network import *
import stream_tee as stream_tee
import __main__ as main
from datetime import datetime

pp = pprint.PrettyPrinter().pprint


def plot_data(data):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)


def get_model_dir():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    name = experiment_name()

    return os.path.join('checkpoints', name) + '/'

def experiment_name():

    experiment = os.path.splitext(os.path.split(main.__file__)[1])[0]
    name = experiment + '_' + stream_tee.generate_timestamp()
    return name


# def preprocess_conf(conf):
#     options = conf.__flags

#     for option, value in options.items():
#         option = option.lower()
#         value = value.value

#         if option == 'hidden_dims':
#             conf.hidden_dims = eval(conf.hidden_dims)
#         elif option == 'w_reg':
#             if value == 'l1':
#                 w_reg = l1_regularizer(conf.w_reg_scale)
#             elif value == 'l2':
#                 w_reg = l2_regularizer(conf.w_reg_scale)
#             elif value == 'none':
#                 w_reg = None
#             else:
#                 raise ValueError('Wrong weight regularizer %s: %s' % (option, value))
#             conf.w_reg = w_reg
#         elif option.endswith('_w'):
#             if value == 'uniform_small':
#                 weights_initializer = random_uniform_small
#             elif value == 'uniform_big':
#                 weights_initializer = random_uniform_big
#             elif value == 'he':
#                 weights_initializer = he_uniform
#             else:
#                 raise ValueError('Wrong %s: %s' % (option, value))
#             setattr(conf, option, weights_initializer)
#         elif option.endswith('_fn'):
#             if value == 'tanh':
#                 activation_fn = tf.nn.tanh
#             elif value == 'relu':
#                 activation_fn = tf.nn.relu
#             elif value == 'none':
#                 activation_fn = None
#             else:
#                 raise ValueError('Wrong %s: %s' % (option, value))
#             setattr(conf, option, activation_fn)


def write_mat(dir, data_dict,filename):
    filename=filename+'.mat'
    timestamp = datetime.now()
    str_time = timestamp.strftime('_%H_%M')
    name='experiment'+str_time
    folder_name = dir
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename=os.path.join(folder_name, filename)
    with open(filename, 'wb') as f:
        scipy.io.savemat(f, data_dict)
    print("Printed .mat files in "+folder_name)


