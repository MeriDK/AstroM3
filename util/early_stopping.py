class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def step(self, metric):
        if self.best_score is None:
            self.best_score = metric
            self.counter = 1
        else:
            if metric < self.best_score:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
        return self.counter >= self.patience
