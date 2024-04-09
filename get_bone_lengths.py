import leap as lp
import json
from utils import (
    get_bone_lengths,
)


class TrackingListener(lp.Listener):
    def __init__(self, callback):
        self.callback = callback

    def on_tracking_event(self, event):
        self.callback(event)


if __name__ == "__main__":

    def callback(data):
        if not data.hands:
            return

        hand = data.hands[0]

        bone_lengths = get_bone_lengths(hand)

        print(bone_lengths)

        with open("bone_lengths.json", "w") as f:
            json.dump(bone_lengths, f)

        exit()

    leap_connection = lp.Connection()

    leap_connection.set_tracking_mode(lp.TrackingMode.Desktop)
    leap_connection.add_listener(TrackingListener(callback))

    with leap_connection.open():
        leap_connection._poll_loop()
