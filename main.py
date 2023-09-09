#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.sparse as sp
import os
import utils
import models
import procedure


tf.flags.DEFINE_string("device_id", '0', "GPU device is to be used in training (default: 0)")
tf.flags.DEFINE_integer("seed", 2, "Seed value for reproducibility (default: 2)")

tf.flags.DEFINE_string("data_dir", './data/tafeng', "The input data directory")
tf.flags.DEFINE_string("output_dir", './output/tafeng/bpcl', "The output directory")

tf.flags.DEFINE_integer("emb_dim", 32, "The dimensionality of embedding (default: 32)")
tf.flags.DEFINE_integer("rnn_unit", 32, "The number of hidden units of RNN (default: 32)")
tf.flags.DEFINE_integer("nb_hop", 1, "The number of neighbor hops  (default: 1)")
tf.flags.DEFINE_integer("nb_epoch", 30, "Number of epochs (default: 30)")
tf.flags.DEFINE_integer("early_stopping_k", 5, "Early stopping patience (default: 5)")
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate (default: 0.01)")
tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout keep probability for RNN (default: 0.3)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size (default: 32)")
tf.flags.DEFINE_integer("display_step", 10, "Show loss/acc for every display_step batches (default: 10)")
tf.flags.DEFINE_string("rnn_cell_type", "LSTM", " RNN Cell Type like LSTM, GRU, etc. (default: LSTM)")
tf.flags.DEFINE_integer("top_k", 10, "Top K Accuracy (default: 10)")

config = tf.flags.FLAGS
print("---------------------------------------------------")
print("SeedVal = " + str(config.seed))
print("\nParameters: " + str(config.__len__()))
for iterVal in config.__iter__():
    print(" + {}={}".format(iterVal, config.__getattr__(iterVal)))
print("Tensorflow version: ", tf.__version__)
print("---------------------------------------------------")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

np.random.seed(config.seed)
tf.compat.v1.set_random_seed(config.seed)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.log_device_placement = False

# ----------------------- MAIN PROGRAM -----------------------

data_dir = config.data_dir
output_dir = config.output_dir

training_file = data_dir + "/train.txt"
validate_file = data_dir + "/validate.txt"
test_file = data_dir + "/test.txt"
print("Output Dir: " + output_dir)

# Create directories
print("@Create directories")
utils.create_folder(output_dir + "/models")

# Load train, validate & test
print("@Load train,validate&test data")
training_instances = utils.read_file_as_lines(training_file)
nb_train = len(training_instances)
print(" + Total training sequences: ", nb_train)

validate_instances = utils.read_file_as_lines(validate_file)
nb_validate = len(validate_instances)
print(" + Total validating sequences: ", nb_validate)

test_instances = utils.read_file_as_lines(test_file)
nb_test = len(test_instances)
print(" + Total test sequences: ", nb_test)

# Create dictionary
print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, rev_item_dict = utils.build_knowledge(training_instances)


print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

print("@Load the normalized adjacency matrix")
matrix_fpath = data_dir + "/adj_matrix/r_matrix_" + str(config.nb_hop)+ "w.npz"
adj_matrix = sp.load_npz(matrix_fpath)
print(" + Real adj_matrix has been loaded from" + matrix_fpath)


print("@Compute #batches in train/validation/test")
total_train_batches = utils.compute_total_batches(nb_train, config.batch_size)
total_validate_batches = utils.compute_total_batches(nb_validate, config.batch_size)
total_test_batches = utils.compute_total_batches(nb_test, config.batch_size)
print(" + #batches in train ", total_train_batches)
print(" + #batches in validate ", total_validate_batches)
print(" + #batches in test ", total_test_batches)

model_dir = output_dir + "/models"
train_logdir = output_dir + "/train_log.txt"

with tf.Session(config=gpu_config) as sess:
    train_generator = utils.seq_batch_generator(training_instances,MAX_SEQ_LENGTH, item_dict, config.batch_size)
    validate_generator = utils.seq_batch_generator(validate_instances, MAX_SEQ_LENGTH,item_dict, config.batch_size, False)
    test_generator = utils.seq_batch_generator(test_instances, MAX_SEQ_LENGTH,item_dict, config.batch_size, False)
    
    print(" + Initialize the network")
    net = models.Beacon(sess, config.emb_dim, config.rnn_unit, MAX_SEQ_LENGTH, adj_matrix, 
                         config.batch_size, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)
    print(" + Initialize parameters")
    sess.run(tf.global_variables_initializer())
    print("================== TRAINING ====================")
    procedure.train_network(sess, net, train_generator, validate_generator, config.nb_epoch,
                            total_train_batches, total_validate_batches, config.display_step,
                            config.early_stopping_k, model_dir,
                            test_generator, total_test_batches, train_logdir)
tf.reset_default_graph()