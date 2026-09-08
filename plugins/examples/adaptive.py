import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from sixseven_plugin import serve


class Model:

    def __init__(self, state):
        self.state = state

    def generate(self, limit):
        if self.state["waiting"]:
            return {"addresses": [], "state": "awaiting_feedback"}
        address = f"2001:db8::{self.state['next']:x}"
        self.state["waiting"] = True
        return {"addresses": [address], "state": "awaiting_feedback"}

    def apply_feedback(self, items):
        observations = [item for item in items if isinstance(item, dict)]
        if observations:
            self.state["next"] += 1
            self.state["waiting"] = False

    def save(self):
        return json.dumps(self.state).encode()


class Algorithm:
    def describe(self):
        return {"id": "example/adaptive", "description": "Adaptive plugin example", "model_version": 1}

    def train(self, config, observations):
        return Model({"next": config.get("start", 1), "waiting": False})

    def load(self, version, payload):
        if version != 1:
            raise ValueError("unsupported model version")
        return Model(json.loads(payload))


if __name__ == "__main__":
    serve(Algorithm())
