def get_network(network_name):
    network_name = network_name.lower()
    # Original CNN
    if network_name == 'model1':
        from .model1 import GenerativeResnet
        return GenerativeResnet
    # Configurable CNN with multiple dropouts
    elif network_name == 'model2':
        from .model2 import GenerativeResnet
        return GenerativeResnet
    # Configurable CNN with dropout at the end
    elif network_name == 'model3':
        from .model4 import GenerativeResnet
        return GenerativeResnet
    # Inverted CNN
    elif network_name == 'model4':
        from .model3 import GenerativeResnet
        return GenerativeResnet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
