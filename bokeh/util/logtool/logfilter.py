class LogFilter():
    def __init__(self, e=3, i=5):
        print("filter init")
        self.e = e
        self.i = i

    def filter(self, record):
        if hasattr(record, "status"):
            record.msg = self.GetLogMsg(record.status)
        return True
    
    def GetLogMsg(self, status):
        msg = f'[epoch: {status["cur_epoch"]:>{self.e}} / {status["max_epoch"]:>{self.e}}]\t\t[itr: {status["cur_itr"]:>{self.i}} / {status["max_itr"]:>{self.i}}]\t\t[ lr: {status["lr"]:.3e} | loss: {status["loss"]:>.6f}'
        return msg
    
    def SetDigits(self, e, i):
        self.e = len(str(e))
        self.i = len(str(i))