from numpy import mean

FILL = "\033[7m"
CLOSE = "\033[0m"

class CLIFilter():

    def __init__(self, length=40):
        print("CLIFilter init")
        self.length = length
        self.losses = []

    def filter(self, record):
        if hasattr(record, "status"):
            record.msg = self.GetCLIMsg(record.status)
        return True
    
    def GetCLIMsg(self, status):
        self.losses.append(status["loss"])
        epoch_bar = self.length * status["cur_epoch"] // status["max_epoch"]
        itr_bar = self.length * status["cur_itr"] // status["max_itr"]

        msg = ("\n[epoc]\t|" + FILL + ("_" * epoch_bar) + CLOSE + "_" * (self.length - epoch_bar) + "|\t\n"
               + "[iter]\t|" + FILL + ("_" * itr_bar) + CLOSE + "_" * (self.length - itr_bar) + "|\n"
               + f"[parm] lr : {status['lr']:<12f}\n"
               + f"[loss] curr : {status['loss']:<12f} mean : {mean(self.losses):<12f}\n"
               + f"[best]")
        
        if itr_bar == self.length:
            self.losses.clear()

        return msg