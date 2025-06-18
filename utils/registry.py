# utils/registry.py

"""
Module registry：维护字符串名称与模块类的映射。
子模块实现类使用 @register_module 装饰器注册自己。
"""

# 全局注册表字典，key 为类名字符串，value 为类对象
REGISTERED_MODULES = {}

def register_module(cls):
    """
    装饰器：将类注册到全局注册表中。
    类名（cls.__name__）会作为键存入 REGISTERED_MODULES。
    """
    name = cls.__name__
    REGISTERED_MODULES[name] = cls
    return cls

def get_module(name):
    """
    根据类名字符串获取注册表中的类对象。
    如果名称不存在会抛出 KeyError。
    """
    try:
        return REGISTERED_MODULES[name]
    except KeyError:
        raise KeyError(f"Module '{name}' is not registered in the registry.")
