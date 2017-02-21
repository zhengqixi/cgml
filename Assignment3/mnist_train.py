from poopoonet import poopoonet
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import json
mnist = read_data_sets("mnist_data/",  one_hot=True)
train = mnist[0]
validate = mnist[1]
test = mnist[2]

train_dict = {'images':train.images, 'labels':train.labels}
validate_dict = {'images':validate.images, 'labels':validate.labels}
test_dict = {'images':test.images, 'labels':test.labels}
net = poopoonet()
net.build_model([28, 28, 1], 10)
epoch, losses, validation_epoch, validation_accuracy = net.train_model(0.003, train_data=train_dict, keep_prob=0.25, validation_data=validate_dict, validation_epoch=1000, outdir="mnist_logdir")
predictions = net.infer_model(test_dict)
total_accuracy= net.accuracy(predictions, test.labels)

with open("validation_data.json", "w") as validation_file:
    json.dump({'epoch':validation_epoch, 'accuracy':validation_accuracy}, validation_file)

with open("loss_data.json", "w") as loss_file:
    json.dump({'epoch':epoch, 'losses':losses}, loss_file)

print(total_accuracy)
