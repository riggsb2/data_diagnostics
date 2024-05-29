import json

def logErrors():
    '''
    Wrapper used on each call function which returns errors found within a function

    :return: exception as string if function raises an error
    '''
    def decorate(f):
        def applicator(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print(e)
                return str(e)
        return applicator
    return decorate


class NpEncoder(json.JSONEncoder):
    """
    Encoder to export objects that include numpy object types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, type(pd.NaT)):
            return None
        elif isinstance(obj, dt) or isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return super(NpEncoder, self).default(obj)
