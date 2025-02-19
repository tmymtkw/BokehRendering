from logging import Logger

class StatusLogger(Logger):
    def __init__(self, 
                 name, 
                 level=0,
                 epochs=1,
                 iteration=10):
        super().__init__(name, level)

    def info(self, msg, *args, exc_info = None, stack_info = False, stacklevel = 1, extra = None):
        
        return super().info(msg, *args, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)

    def GetStatusMsg():
        pass