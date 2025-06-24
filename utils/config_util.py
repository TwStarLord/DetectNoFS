import yaml

def load_config(config_path):
    """
    从 YAML 配置文件加载配置并返回字典。
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
