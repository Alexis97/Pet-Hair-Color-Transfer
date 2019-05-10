import tensorflow as tf
import os


class Logger(object):
	"""Tensorboard logger."""

	def __init__(self, opt):
		"""Initialize summary writer."""
		if not os.path.exists(opt.log_dir):
			os.makedirs(opt.log_dir)
		self.writer = tf.summary.FileWriter(opt.log_dir)

	def scalar_summary(self, tag, value, step):
		"""Add scalar summary."""
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
		self.writer.add_summary(summary, step)