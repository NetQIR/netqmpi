import runpy

def load_main(path):
    if path is None:
        raise ValueError("script must be provided")
    if not path.endswith(".py"):
        raise ValueError("script must be a .py script")
    
    namespace = runpy.run_path(path)

    if "main" not in namespace:
        raise ValueError(f"{path} does not define a main() function")

    if namespace["main"] is None:
            raise ValueError(f"main function not found in {path}")

    return namespace["main"]