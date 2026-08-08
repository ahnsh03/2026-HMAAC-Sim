import os
import types
import marshal

def get_path(file_name=None):
    # Load libs next to this file (works under /root/hmobility_ws and symlink-install)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

def get_pyc(module_file):
    file_path = get_path(module_file)
    print('\n파일명:', file_path, '\n\n')
    pyc = open(file_path, 'rb').read()
    code = marshal.loads(pyc[16:])
    module = types.ModuleType('module_name')
    exec(code, module.__dict__)
    return module

decision_making_func_lib = get_pyc("decision_making_func_lib.cpython-310.pyc")
