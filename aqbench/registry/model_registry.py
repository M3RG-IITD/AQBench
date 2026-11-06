MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name):
    print("Registered models:", MODEL_REGISTRY.keys())
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not registered")

    return MODEL_REGISTRY[name]

