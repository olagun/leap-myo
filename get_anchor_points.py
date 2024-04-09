import leap as lp
import json
import numpy as np
from utils import (
    get_anchor_points,
)


class TrackingListener(lp.Listener):
    def __init__(self, callback):
        self.callback = callback

    def on_tracking_event(self, event):
        self.callback(event)


class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":

    def callback(data):
        if not data.hands:
            return

        hand = data.hands[0]

        anchor_points = get_anchor_points(hand)

        print(anchor_points)

        with open("anchor_points.json", "w") as f:
            json.dump(anchor_points, f, cls=json_serialize)

        exit()

    leap_connection = lp.Connection()

    leap_connection.set_tracking_mode(lp.TrackingMode.Desktop)
    leap_connection.add_listener(TrackingListener(callback))

    with leap_connection.open():
        leap_connection._poll_loop()
