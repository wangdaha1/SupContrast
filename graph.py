
# this is used for drawing graphs by extracting data from tensorboard records
import ssl

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

SSL=event_accumulator.EventAccumulator(
    r'./save/noisy_FT_1/ssl/cifar100_tensorboard/pairflip_0.4_cifar100_resnet18_try1/valid/events.out.tfevents.1643097091.yuany3.math.ust.hk.1759.1')
SSL.Reload()

SL = event_accumulator.EventAccumulator(
    r'./save/noisy_FT_1/sl/cifar100_tensorboard/pairflip_0.4_cifar100_resnet18_try1/valid/events.out.tfevents.1644296827.yuany2.math.ust.hk.6717.1')
SL.Reload()

FS = event_accumulator.EventAccumulator(
r'./save/noisy_CE/cifar100_tensorboard/pairflip_0.4_cifar100_resnet18_try1/valid/events.out.tfevents.1644164058.yuany2.math.ust.hk.7723.1'
)
FS.Reload()

data_ssl = SSL.scalars.Items('acc')
data_sl = SL.scalars.Items('acc')
data_fs = FS.scalars.Items('acc')
plt.plot([i.step for i in data_ssl],[i.value for i in data_ssl], label='ssl')
plt.plot([i.step for i in data_sl],[i.value for i in data_sl], label='nsl')
plt.plot([i.step for i in data_fs],[i.value for i in data_fs], label='from scratch')
plt.ylabel('acc')
plt.legend(loc='upper right')
plt.show()
plt.savefig('./figures/pairflip_0.4_cifar100_resnet18_FT_1(valid)_try1.png', bbox_inches='tight')

#
# Clean = event_accumulator.EventAccumulator(
#     r'./save/clean_FT/ssl/cifar10_tensorboard/cifar10_resnet18_ssl_pretrained/train/events.out.tfevents.1642830478.yuany3.math.ust.hk.27309.0')
# Clean.Reload()
#
# Noisy = event_accumulator.EventAccumulator(
# r'./save/noisy_FT_1/ssl/cifar10_tensorboard/symmetric_0.4_cifar10_resnet18_try1/train/events.out.tfevents.1642840796.yuany3.math.ust.hk.24640.0')
# Noisy.Reload()
#
# data_clean = Clean.scalars.Items('loss')
# data_noisy = Noisy.scalars.Items('loss')
# plt.plot([i.step for i in data_clean],[i.value for i in data_clean], label='clean')
# plt.plot([i.step for i in data_noisy],[i.value for i in data_noisy], label='noisy')
# plt.ylabel('loss')
# plt.legend(loc='lower right')
# plt.show()
# plt.savefig('./figures/clean_noisy_SSL_loss.png', bbox_inches='tight')



# print(SSL.scalars.Keys())
# SSL_label_precision_rate = SSL.scalars.Items('label_precision_rate')
# SSL_clean_selection_num = SSL.scalars.Items('clean_selection_num')
# SSL_noise_detection_rate = SSL.scalars.Items('noise_detection_rate')
# SSL_noise_detect_num = SSL.scalars.Items('noise_detect_num')
# SL_label_precision_rate = SL.scalars.Items('label_precision_rate')
# SL_clean_selection_num = SL.scalars.Items('clean_selection_num')
# SL_noise_detection_rate = SL.scalars.Items('noise_detection_rate')
# SL_noise_detect_num = SL.scalars.Items('noise_detect_num')
#
# # draw graphs
# plt.subplot(221)
# data_ssl = SSL_label_precision_rate
# data_sl = SL_label_precision_rate
# plt.plot([i.step for i in data_ssl],[i.value for i in data_ssl], label='ssl')
# plt.plot([i.step for i in data_sl],[i.value for i in data_sl], label='sl')
# plt.ylabel('label_precision_rate')
# plt.legend(loc='upper right')
# plt.show()
#
# plt.subplot(222)
# data_ssl = SSL_clean_selection_num
# data_sl = SL_clean_selection_num
# plt.plot([i.step for i in data_ssl],[i.value for i in data_ssl], label='ssl')
# plt.plot([i.step for i in data_sl],[i.value for i in data_sl], label='sl')
# plt.ylabel('clean_selection_num')
# plt.legend(loc='upper right')
# plt.show()
#
# plt.subplot(223)
# data_ssl = SSL_noise_detection_rate
# data_sl = SL_noise_detection_rate
# plt.plot([i.step for i in data_ssl],[i.value for i in data_ssl], label='ssl')
# plt.plot([i.step for i in data_sl],[i.value for i in data_sl], label='sl')
# plt.ylabel('noise_detection_rate')
# plt.legend(loc='upper right')
# plt.show()
#
# plt.subplot(224)
# data_ssl = SSL_noise_detect_num
# data_sl = SL_noise_detect_num
# plt.plot([i.step for i in data_ssl],[i.value for i in data_ssl], label='ssl')
# plt.plot([i.step for i in data_sl],[i.value for i in data_sl], label='sl')
# plt.ylabel('noise_detect_num')
# plt.legend(loc='upper right')
# plt.show()
#
# plt.savefig('./figures/pairflip_0.4_cifar10_resnet18_clean_FT(train)_try1.png', bbox_inches='tight')

