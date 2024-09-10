class LRLambda:
    def __init__(self, n_epochs, offset, decay_start):
        assert (
            n_epochs - decay_start
        ) > 0, "Decay Needs to start Before Epoch Runs finish"

        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start = decay_start

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start) / (
            self.n_epochs - self.decay_start
        )
