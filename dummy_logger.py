from logging import Logger

class DummyLogger(Logger):

    def error(self, msg, *args, exc_info = None, stack_info = False, stacklevel = 1, extra = None):
        return None
    
    def info(self, msg, *args, exc_info = None, stack_info = False, stacklevel = 1, extra = None):
        return None