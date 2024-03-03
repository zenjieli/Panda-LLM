from threading import Event
import utils.text_processing as text_processing

class BaseModel:
    def __init__(self) -> None:
        self.stop_event = Event()

    def predict(self):
        pass

    def support_image(self):
        return False

    def append_user_input(self, query, history, task_history):
        if history is None:
            history = []

        if query != '':
            return '', history + [[text_processing.parse_text(query), '']], task_history
        else:
            return '', history, task_history