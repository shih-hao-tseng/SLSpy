import inspect
class ObjBase:
    '''
    The object base that defines debugging tools
    '''
    def initialize (self, **kwargs):
        pass

    def sanityCheck (self):
        # check the system parameters are coherent
        return True

    def errorMessage (self,msg):
        print(self.__class__.__name__+'-'+inspect.stack()[1][3]+': [ERROR] '+msg+'\n')
        return False