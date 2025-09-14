import copy


def model_to_update(global_model, local_model):
    global_model = global_model.state_dict()
    local_update = []
    for key, value in local_model.state_dict().items():
        local_update.append(value - global_model[key])
    return local_update


def update_to_model(global_model, local_update):
    local_model = copy.deepcopy(global_model)
    i = 0
    for key, value in local_model.state_dict().items():
        v_size = value.reshape(-1, 1).size()[0]
        value += local_update[i:(i + v_size)].reshape(value.shape)
        i += v_size
    return local_model


def update_vector_to_array(model, update_vector):
    update_array = []
    i = 0
    for key, value in model.state_dict().items():
        v_size = value.reshape(-1, 1).size()[0]
        update_list = update_vector[i:(i + v_size)].reshape(value.shape)
        update_array.append(update_list.cpu().numpy())
        i += v_size
    return update_array


def layer_size(model):
    size = []
    for key, value in model.state_dict().items():
        size.append(value.reshape(-1, 1).size()[0])
    return size
