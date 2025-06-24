# import pkgutil
# import importlib
# import blocks
#
# def register_all():
#     """
#     遍历 blocks 包下的所有子模块并导入它们，以触发注册装饰器。
#     """
#     # pkgutil.walk_packages 会导入 packages，以查找其 __path__ 中的子模块
#     package_dir = blocks.__path__
#     prefix = blocks.__name__ + "."
#     for _, module_name, _ in pkgutil.walk_packages(package_dir, prefix):
#         m = importlib.import_module(module_name)
#
#
# if __name__ == "__main__":
#     # 执行注册：导入所有模块
#     register_all()
