import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.summary import Logger
import tensorflow as tf

if __name__ == '__main__':
	opt = TrainOptions().parse()
	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)

	model = create_model(opt)
	model.setup(opt)
	# visualizer = Visualizer(opt)

	# Create summary writer
	logger = Logger(opt)
	tags = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
	# loss_sum = dict.fromkeys(tags, 0)
	# loss_avg = dict.fromkeys(tags, 0)

	total_steps = 0
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()
		iter_data_time = time.time()
		epoch_iter = 0

		for i, data in enumerate(dataset):
			iter_start_time = time.time()
			if total_steps % opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time
			# visualizer.reset()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters()

			# losses = model.get_current_losses()
			# for tag in tags:
			# 	loss_sum[tag] += losses[tag]
			# 	loss_avg[tag] = loss_sum[tag] / total_steps

			if total_steps % opt.display_freq == 0:
				save_result = total_steps % opt.update_html_freq == 0
				# visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

			if total_steps % opt.print_freq == 0:
				losses = model.get_current_losses()
				t = (time.time() - iter_start_time) / opt.batchSize

				print ('step %d, cost %.3f secs, D_A:%.3f, G_A:%.3f, cycle_A:%.3f, idt_A:%.3f, D_B:%.3f, G_B:%.3f, cycle_B:%.3f, idt_B:%.3f ' %
				 (total_steps, t, losses['D_A'], losses['G_A'], losses['cycle_A'], losses['idt_A'], losses['D_B'], losses['G_B'], losses['cycle_B'], losses['idt_B'] ))
				
				
				for tag in tags:
					logger.scalar_summary(tag, losses[tag], total_steps)
					
				# visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
				# if opt.display_id > 0:
					# visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %
					  (epoch, total_steps))
				model.save_networks('latest')

			iter_data_time = time.time()
		if epoch % opt.save_epoch_freq == 0:

			# print ('Average loss:  D_A:%.3f, G_A:%.3f, cycle_A:%.3f, idt_A:%.3f, D_B:%.3f, G_B:%.3f, cycle_B:%.3f, idt_B:%.3f ' %
			# 	 (total_steps, t, loss_avg['D_A'], loss_avg['G_A'], loss_avg['cycle_A'], loss_avg['idt_A'], loss_avg['D_B'], loss_avg['G_B'], loss_avg['cycle_B'], loss_avg['idt_B'] ))
				
			print('saving the model at the end of epoch %d, iters %d' %
				  (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
			  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
		model.update_learning_rate()
