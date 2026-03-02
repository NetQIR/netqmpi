import runpy

def load_main(path):
    namespace = runpy.run_path(path)

    if "main" not in namespace:
        raise ValueError(f"{path} does not define a main() function")

    return namespace["main"]