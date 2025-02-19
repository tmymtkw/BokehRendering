import os.path
import json

class ConfigLoader(object):
    def __init__(self, path:str = "GeneralLibrary/config/defaul.json"):
        assert os.path.exists(path), self.PutError(path)

        with open(path, "r") as f:
            self.cfg = json.load(f)

        print(self.cfg, end="\n\n")

    def GetInfo(self, category, name):
        return self.cfg[category][name]

    def GetPath(self, name):
        return self.GetInfo("path", name)
    
    def GetHyperParam(self, name):
        return self.GetInfo("hyperparam", name)
        # return self.cfg["hyperparam"][name]

    def PutError(self, path):
        return f"\n[Error] ファイルが見つかりません \
                 \nInput: {path} \
                 \nAbs path: {os.path.abspath(path)}\n"
    
