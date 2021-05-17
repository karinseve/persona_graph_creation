
class DictQuery(dict):
    """Something to traverse complex JSON structures easier."""

    def __init__(self, *args, **kwargs):
        """If converting a dict to DictQuery, any dictionaries held as values
        are recursively converted to DictQuery as well."""
        super().__init__(*args, **kwargs)
        for key in self.keys():
            if isinstance(self[key], dict):
                self[key] = DictQuery(self[key])

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def get(self, p, default=None):
        keys = p.split(".")
        val = self
        for key in keys:
            if isinstance(val, list):
                val = [v.get(key) if v else None for v in val]
            elif isinstance(val, dict):
                val = dict.get(val, key)
            else:  # not a dict and we still have more keys to go
                return default
            if val is None:  # wrong key
                return default
        return val

    def search_key(self, key):
        if key in self.keys():
            return self[key]
        for item in self.items():
            if isinstance(item, tuple):
                item = DictQuery(item)
                item.search_key(key)
