import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.sparse as sp
import utils

# Model hyper-parameters
tf.flags.DEFINE_string("data_dir", './data/beauty', "The input data directory")
tf.flags.DEFINE_integer("nb_hop", 1, "The order of the real adjacency matrix (default:1)")

config = tf.flags.FLAGS
print("---------------------------------------------------")
print("Data_dir = " + str(config.data_dir))
print("\nParameters: " + str(config.__len__()))
for iterVal in config.__iter__():
    print(" + {}={}".format(iterVal, config.__getattr__(iterVal)))
print("Tensorflow version: ", tf.__version__)
print("---------------------------------------------------")

# ----------------------- MAIN PROGRAM -----------------------
data_dir = config.data_dir
output_dir = data_dir + "/adj_matrix"

training_file = data_dir + "/train.txt"
validate_file = data_dir + "/validate.txt"
testing_file = data_dir + "/test.txt"


utils.create_folder(output_dir)
training_instances = utils.read_file_as_lines(training_file)
nb_train = len(training_instances)
print(" + Total training sequences: ", nb_train)

print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict = utils.build_knowledge(training_instances)

rmatrix_fpath = output_dir + "/r_matrix_" + str(config.nb_hop) + "w.npz"
real_adj_matrix = utils.build_sparse_adjacency_matrix_v2(training_instances, item_dict)
real_adj_matrix = utils.normalize_adj(real_adj_matrix)
real_adj_matrix = utils.remove_diag(real_adj_matrix)

mul = real_adj_matrix
with tf.device('/cpu:0'):
    w_mul = real_adj_matrix
    coeff = 1.0
    for w in range(1, config.nb_hop):
        coeff *= 0.85
        w_mul *= real_adj_matrix
        w_mul = utils.remove_diag(w_mul)

        w_adj_matrix = utils.normalize_adj(w_mul)
        mul += coeff * w_adj_matrix

    real_adj_matrix = mul

    sp.save_npz(rmatrix_fpath, real_adj_matrix)
    print(" + Save adj_matrix to" + rmatrix_fpath)
