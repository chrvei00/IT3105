class TestPlant:
    def __init__(self):
        self.process_value = 0
        self.time = 0
        self.max_time = 100
        self.max_process_value = 100
        self.min_process_value = 0
        self.process_value_step = 1

    def update(self):
        self.time += 1
        self.process_value += self.process_value_step

    def apply_control_action(self, control_action):
        self.process_value_step = control_action

    def get_process_value(self):
        return self.process_value

    def get_time(self):
        return self.time

    def get_max_time(self):
        return self.max_time

    def get_max_process_value(self):
        return self.max_process_value

    def get_min_process_value(self):
        return self.min_process_value

    def get_process_value_step(self):
        return self.process_value_step