import os   
import json
import numpy as np
from snn_delays.config import CHECKPOINT_PATH


def save_to_numpy(snn, directory='default'):
    """
    Function to save weights and biases in numpy format. This function
    permit to save these parameters to be loaded in a device like
    SpiNNaker.

    :param snn: The network to save.
    :param directory: Directory to save the model (relative to
    CHECKPOINT_PATH) (default = 'default')
    """

    # Define the path to save the layer weights and biases
    layers_path = os.path.join(CHECKPOINT_PATH, directory)

    # If the directory do not exist, it is created
    if not os.path.isdir(layers_path):
        os.mkdir(layers_path)

    # Define the sub-path with the name of the model
    layers_sub_path = os.path.join(str(layers_path),
                                    'model_' + str(snn.model_name))

    # If the directory do not exist, it is created
    if not os.path.isdir(layers_sub_path):
        os.mkdir(layers_sub_path)

    # Initialize weight-and-biases and the state dictionary of the network
    weights_biases = []
    snn_state_dict = snn.state_dict()

    # Open a file to write info about the model
    with open(layers_sub_path + '/model_info', 'a') as logs:
        # spk = snn.test_spk_count[-1][1].detach().cpu().numpy()
        spk = snn.test_spk_count[-1][1]
        logs.write("avg spk neuron/sample {}".format(spk))
        logs.write("\navg spk neuron/timestep {}".format(
            spk * (snn.num_hidden / snn.win)))

    # Save several arrays into a single file in uncompressed .npz format
    for k in snn_state_dict:
        np.savez(layers_sub_path + '/' + k,
                    snn_state_dict[k].data.cpu().numpy())
        weights_biases.append(snn_state_dict[k].data.cpu().numpy())

    print('Weights and biases saved in ', layers_sub_path)


# TODO: ¿Crear un metodo mas general para que sirva en redes con y sin
#  delay?
#  Documentar
#  No funciona la opcion multi_delays por el cambio de nombres de las capas
#  cuando se reescribe el metodo set_input_layer
def save_to_json_list(snn, directory='default', model_name='rsnn',
                        multi_delays=True):
    """
    Function to save in a json file the list ... ?¿

    :param directory: Directory to save the figure (default = 'default')
    :param model_name: Model name of the network (default = 'rsnn')
    :param multi_delays: ¿?
    """

    # Set directory to save the json file
    layers_location = os.path.join(CHECKPOINT_PATH, directory,
                                    'model_' + str(model_name))

    # If the directory do not exist, it is created
    if not os.path.isdir(layers_location):
        os.mkdir(layers_location)

    # Initialization of the dictionary to save in json file
    weight_delay_dict = {}

    #
    # if multi_delays:
    #     weights_ih = [
    #         getattr(snn, 'f0_id' +
    #                 str(d)).weight.data.detach().cpu().numpy()
    #                 for d in snn.delays]
    #     inh, exc = snn.project_ih_weights(weights_ih, snn.delays)
    # else:
    weights_ih = snn.f0_i.weight.data.detach().cpu().numpy()
    inh, exc = snn.project_weights(weights_ih)

    weight_delay_dict['f0_i'] = {'exc': exc, 'inh': inh}

    #
    for name in snn.h_names:
        h_weights = getattr(snn, name).weight.data.detach().cpu().numpy()
        inh, exc = snn.project_weights(h_weights)
        weight_delay_dict[name] = {'exc': exc, 'inh': inh}

    # Save dictionary
    dict_name = os.path.join(layers_location, str(snn.dataset[:-3]))
    with open("{}.json".format(dict_name), 'w') as outfile:
        json.dump(weight_delay_dict, outfile)

# TODO: Documentar, ¿que hace?
def project_ih_weights(weights, delays):
    """
    Auxiliary function to the method save_to_json_list.

    This function project ... ¿?
    """

    # Initialization of inhibitory and excitatory synapses
    inh_synapses = []
    exc_synapses = []

    # Loop over the weights
    for wi, w in enumerate(weights):
        for i in range(w.shape[1]):
            for j in range(w.shape[0]):
                if float(w[j, i]) != 0.0:
                    if float(w[j, i]) < 0.0:
                        inh_synapses.append(
                            [i, j, float(-1.0 * w[j, i]),
                                int(delays[wi] + 1)])
                    else:
                        exc_synapses.append([i, j, float(w[j, i]),
                                                int(delays[wi] + 1)])
    return inh_synapses, exc_synapses

# TODO: Documentar, ¿que hace?
def project_weights(weights, delay=0):
    """
    Auxiliary function to the method save_to_json_list.

    This function project ... ¿?
    """

    # Initialization of inhibitory and excitatory synapses
    inh_synapses = []
    exc_synapses = []

    # Loop over the weights
    for i in range(weights.shape[1]):
        for j in range(weights.shape[0]):
            if float(weights[j, i]) < 0.0:
                inh_synapses.append(
                    [i, j, float(-1.0 * weights[j, i]), delay + 1])
            else:
                exc_synapses.append(
                    [i, j, float(weights[j, i]), delay + 1])

    return inh_synapses, exc_synapses