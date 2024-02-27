from threading import Event

class BaseModel:
    def __init__(self) -> None:
        self.stop_event = Event()

    def predict(self):
        pass