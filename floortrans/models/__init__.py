from floortrans.models.hg_furukawa_original import *

def get_model(folder_path,name,n_classes=None, version=None):
    if name == 'hg_furukawa_original':
        model = hg_furukawa_original(folder_path=folder_path,n_classes=n_classes)
        model.init_weights()
    else:
        raise ValueError('Model {} not available'.format(name))

    return model


