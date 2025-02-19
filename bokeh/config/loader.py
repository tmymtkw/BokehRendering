import os.path
import json

class ConfigLoader(object):
    def __init__(self, path:str = "GeneralLibrary/config/defaul.json"):
        print("ConfigLoader initilalizing...")
        assert(os.path.exists(path), self.PutError(path))

    def PutError(self, path) -> str:
        return f"[Error] file not found\nInput: {path}\nAbs path: {os.path.abspath(path)}\n"