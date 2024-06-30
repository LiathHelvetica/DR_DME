from torch.optim.lr_scheduler import LRScheduler


class NoOpScheduler(LRScheduler):
	def __init__(self):
		pass

	def step(self):
		pass
