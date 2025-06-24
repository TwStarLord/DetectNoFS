# register.py

import importlib
import pkgutil
import inspect
import blocks

class ModuleRegister:
    """
    模块注册器：接收 config2D.yaml 中的 blocks 配置字典，
    自动遍历 blocks 包下的所有子模块并注册其中定义的类，
    并提供根据 category（如 'ca', 'DIFB'）从配置中取到类名后实例化的接口。
    """
    def __init__(self, blocks_config: dict):
        """
        :param blocks_config: config2D.yaml 中的 blocks 字典，
            例如：
            {
              "ca": "MutilHeadCrossAttentionBlock",
              "cls_head": "GeMClassifier",
              "DIFB": "DIFB2D",
              "feature_fusion": "AFFFeatureFsuion",
              "upsample": "PatchExpandConvTranspose"
            }
        """
        self.config = blocks_config
        self.module_registry = {}
        self._register_all()

    def _register_all(self):
        """
        遍历 blocks 包下的所有子模块并导入，
        对每个模块中定义的类按名称注册到 self.module_registry。
        """
        pkg_path = blocks.__path__
        prefix   = blocks.__name__ + "."
        for _, module_name, _ in pkgutil.walk_packages(pkg_path, prefix):
            module = importlib.import_module(module_name)
            # 将模块内定义的所有类注册
            for cls_name, cls_obj in inspect.getmembers(module, inspect.isclass):
                if cls_obj.__module__ == module_name:
                    self.module_registry[cls_name] = cls_obj

    def get_module_class(self, class_name: str):
        """
        根据类名字符串获取注册表中的类对象，若不存在则抛出 KeyError。
        """
        if class_name not in self.module_registry:
            raise KeyError(f"No module named '{class_name}' is registered.")
        return self.module_registry[class_name]

    def build_module(self, class_name: str, *args, **kwargs):
        """
        根据类名字符串实例化模块，构造参数透传给类的 __init__。
        """
        cls = self.get_module_class(class_name)
        return cls(*args, **kwargs)

    def build_from_config(self, category: str, *args, **kwargs):
        """
        根据 category（例如 'ca'、'DIFB'）：
          1. 从 self.config[category] 获取实际要用的类名；
          2. 调用 build_module 实例化并返回。
        """
        if category not in self.config:
            raise KeyError(f"Config 中缺少 category='{category}'")
        class_name = self.config[category]
        return self.build_module(class_name, *args, **kwargs)
