import os

OBSTACLES = ['floor']
SURFACES = ['table', 'tray', 'stove', 'button', 'placemat'] # TODO(caelan): refactor

NAME_TEMPLATE = '{}#{}'

##################################################


def is_surface(name):
    return any(surface_type in name for surface_type in SURFACES)


def is_fixed(name):
    return any(surface_type in name for surface_type in OBSTACLES + SURFACES)


def is_item(name):
    return not is_fixed(name)


def get_type(name):
    return name.split('#' if '#' in name else '-')[0]


def get_models_path():
    current_dir = os.path.dirname(os.path.abspath(__file__)) # abspath = realpath
    return os.path.abspath(os.path.join(current_dir, os.pardir, 'models'))


def get_body_urdf(name):
    '''
    Distinct objects should be of form greenblock#1,
        where anything after the # is ignored for selecting the URDF
    '''
    return os.path.join(get_models_path(), '{}.urdf'.format(get_type(name)))


def create_name(ty, num):
    return NAME_TEMPLATE.format(ty, num)
