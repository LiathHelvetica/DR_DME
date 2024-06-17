from numpy import ndarray


class FundusImage:

	def __init__(self, image: ndarray, file_name: str):
		self.image: ndarray = image
		self.file_name: str = file_name
