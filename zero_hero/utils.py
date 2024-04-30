class FakeWandb:
    def init(self, *args, **kwargs) -> None:
        print(kwargs)

    def log(self, logs):
        print(logs)
