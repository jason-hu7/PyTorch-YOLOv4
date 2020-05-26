from typing import List, Dict

def parse_model_config(path: str) -> List[Dict[str, str]]:
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    # Read the config file
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x.rstrip().lstrip() for x in lines if x and not x.startswith('#')]
    # Add each block to a list
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            # Convolutional layers all need to be specified of batch norm key
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


def parse_data_config(path: str) -> Dict[str, str]:
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
