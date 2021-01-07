import ast
import configparser
from os import path, listdir
import io

_CONVERTERS = {
    "struct": ast.literal_eval
}

_DEFAULTS_DIR = path.abspath(path.join(path.split(__file__)[0], "defaults"))
DEFAULTS = dict()

for file in listdir(_DEFAULTS_DIR):
    name, ext = path.splitext(file)
    if ext == ".ini":
        DEFAULTS[name] = path.join(_DEFAULTS_DIR, file)


def load_config(config_file, defaults_file):
    parser = configparser.ConfigParser(allow_no_value=True, converters=_CONVERTERS)
    parser.read([defaults_file, config_file])
    return parser


def config_to_string(config):
    with io.StringIO() as sio:
        config.write(sio)
        config_str = sio.getvalue()
    return config_str
