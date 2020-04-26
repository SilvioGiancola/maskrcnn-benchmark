import threading
import time
from datetime import datetime

def training_in_background(param_train):
    start_training_date = datetime.now().strftime("%H:%M:%S")
    print(f'[TRAINING THREAD - {start_training_date} - {param_train["asked_training_date"]}] START training')
    print(f'[TRAINING THREAD - {start_training_date} - {param_train["asked_training_date"]}] list of images are ')
    print(f'[TRAINING THREAD - {start_training_date} - {param_train["asked_training_date"]}] {param_train["training_images"]}')
    print(f'[TRAINING THREAD - {start_training_date} - {param_train["asked_training_date"]}] {param_train["validation_images"]}')
    time.sleep(param_train["time"])
    print(f'[TRAINING THREAD - {start_training_date} - {param_train["asked_training_date"]}] DONE training after {param_train["time"]}sec')



class TrainingThread(threading.Thread):

    def __init__(self, function_that_train=training_in_background):
        threading.Thread.__init__(self)
        self.train = function_that_train
        self.list_of_trainings_parameters = []
        self.running = False

    def run(self):
        while(True):
            time.sleep(1)
            if len(self.list_of_trainings_parameters) > 0:
                self.running = True
                self.train(self.list_of_trainings_parameters[0])
                self.list_of_trainings_parameters.pop(0)
                self.running = False

    def append_training(self, parameters):
        if parameters["project_ID"] in self.list_project_waiting_training():
            print("[TRAINING THREAD] Training already in queue")
            return True
        self.list_of_trainings_parameters.append(parameters)
        return False

    def list_project_waiting_training(self):
        l = list(set(([p["project_ID"] for p in self.list_of_trainings_parameters])))
        print(l)
        return l
    
    def number_project(self):
        return len(self.list_of_trainings_parameters)

    def currently_running(self):
        return self.running

