class FakeWandb:
    def __init__(self, *args, **kwargs) -> None:
        print(args)
        print(kwargs)

    def log(self, logs):
        print(logs)
