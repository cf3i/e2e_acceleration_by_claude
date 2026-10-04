def make_engine(name: str, **kw):
    if name == "naive":
        from .naive import NaiveEngine
        return NaiveEngine(**kw)
    raise ValueError(f"unknown engine {name}")
