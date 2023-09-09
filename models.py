from torch import log
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from layers import *
from abc import abstractmethod


class Model:

    def __init__(self, sess, seed, learning_rate, name='model'):
        self.scope = name
        self.session = sess
        self.seed = seed
        self.learning_rate = tf.constant(learning_rate)

    @abstractmethod
    def train_batch(self, s, s_length, y):
        pass

    @abstractmethod
    def validate_batch(self, s, s_length, y):
        pass

    @abstractmethod
    def generate_prediction(self, s, s_length):
        pass


class BPCL(Model):

    def __init__(self, sess, emb_dim, rnn_units, max_seq_length, adj_matrix,
                 batch_size, rnn_cell_type, rnn_dropout_rate, seed, learning_rate,beta=0.1):

        super().__init__(sess, seed, learning_rate, name="BPCL")

        self.emb_dim = emb_dim
        self.rnn_units = rnn_units

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.beta=beta

        with tf.name_scope(self.scope):
            self.A = tf.constant(adj_matrix.todense(), name="Adj_Matrix", dtype=tf.float32)
            self.y = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.nb_items), name='Target_basket')      

            with tf.name_scope("Basket_Sequence_Encoder"):
                self.bseq = tf.placeholder(shape=(batch_size, self.max_seq_length, self.nb_items), dtype=tf.float32, name="bseq_input")
                self.bseq_length = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name='bseq_length')
                self.h_T=self.get_h_T(self.bseq,self.bseq_length,emb_dim,rnn_dropout_rate,rnn_cell_type)
                
            with tf.name_scope("Augmentation"):
                bseq_aug1=self.basket_augment(self.bseq,self.emb_dim)
                bseq_aug2=self.basket_augment(self.bseq,self.emb_dim)
                aug1_h_T=self.get_h_T(bseq_aug1,self.bseq_length,emb_dim,rnn_dropout_rate,rnn_cell_type)
                aug2_h_T=self.get_h_T(bseq_aug2,self.bseq_length,emb_dim,rnn_dropout_rate,rnn_cell_type)
            with tf.name_scope("CLoss"):
                self.closs=self.compute_closs(aug1_h_T,aug2_h_T,self.h_T)

            with tf.name_scope("Next_Basket"):
                self.W_H1 = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal((self.rnn_units,self.nb_items),stddev=0.01), name="W_H1")
                logits = tf.matmul(self.h_T,tf.sigmoid(tf.matmul(self.W_H1,self.A)))
               
            with tf.name_scope("RLoss"):
                self.rloss = self.compute_rloss(logits, self.y)
            
            with tf.name_scope("Loss"):
                self.loss=self.rloss+self.beta*self.closs
                self.predictions = tf.nn.sigmoid(logits)
                self.top_k_values, self.top_k_indices = tf.nn.top_k(self.predictions, 200)
                self.quality_at_5 = self.compute_quality_at_topk(5)
                self.quality_at_10 = self.compute_quality_at_topk(10)
                self.quality_at_20 = self.compute_quality_at_topk(20)
                self.quality_at_30 = self.compute_quality_at_topk(30)

            train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            self.grads = train_op.compute_gradients(self.loss, tf.trainable_variables())
            self.update_grads = train_op.apply_gradients(self.grads)


    def train_batch(self, s, s_length, y):
        _,loss, (correct_preds5,actual_bsize5,f1_score5), (correct_preds10,actual_bsize10,f1_score10), (correct_preds20,actual_bsize20,f1_score20),(correct_preds30,actual_bsize30,f1_score30) = self.session.run(
            [self.update_grads,self.loss, self.quality_at_5, self.quality_at_10, self.quality_at_20,self.quality_at_30],
            feed_dict={self.bseq_length: s_length, self.y: y, self.bseq:s})

        return loss, correct_preds5,actual_bsize5,f1_score5, correct_preds10,actual_bsize10,f1_score10,correct_preds20,actual_bsize20,f1_score20,correct_preds30,actual_bsize30,f1_score30

    def validate_batch(self, s, s_length, y):
        loss, (correct_preds5,actual_bsize5,f1_score5), (correct_preds10,actual_bsize10,f1_score10), (correct_preds20,actual_bsize20,f1_score20),(correct_preds30,actual_bsize30,f1_score30)= self.session.run(
            [self.loss, self.quality_at_5, self.quality_at_10, self.quality_at_20,self.quality_at_30],
            feed_dict={ self.bseq_length: s_length, self.y: y, self.bseq:s})
        return loss, correct_preds5,actual_bsize5,f1_score5, correct_preds10,actual_bsize10,f1_score10,correct_preds20,actual_bsize20,f1_score20,correct_preds30,actual_bsize30,f1_score30

    def generate_prediction(self, s, s_length):
         return self.session.run([self.top_k_values, self.top_k_indices],
                                 feed_dict={self.bseq_length: s_length, self.bseq:s})

    def get_item_bias(self):
        return self.session.run(self.I_B)

    def basket_augment(self, binput,emb_dim):
        bseq_aug=tf.reshape(binput, shape=[-1, self.nb_items], name="X_aug")

        mask=tf.where(tf.random_uniform(shape=[self.batch_size*self.max_seq_length, self.nb_items])>0.5,tf.ones(shape=[self.batch_size*self.max_seq_length, self.nb_items]),tf.zeros(shape=[self.batch_size*self.max_seq_length, self.nb_items]))
        bseq_aug=create_basket_encoder(bseq_aug, emb_dim, param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu, name="Basket_aug_Encoder")
        A_t=create_basket_encoder(self.A, emb_dim, param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu, name="Adjacent_Encoder")
        bseq_aug = tf.nn.relu(tf.matmul(bseq_aug, tf.transpose(A_t), name="X_augxA")) 
        bseq_aug=tf.multiply(mask,bseq_aug)
        bseq_aug=tf.reshape(bseq_aug, shape=[-1, self.max_seq_length, self.nb_items], name="bsxMxN")
        return bseq_aug

    def get_h_T(self,bseq,bseq_length,emb_dim,rnn_dropout_rate,rnn_cell_type):
        bseq_encoder = create_basket_encoder(bseq, emb_dim, param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu)       
        rnn_encoder = create_rnn_encoder(bseq_encoder, self.rnn_units, rnn_dropout_rate, self.bseq_length, rnn_cell_type, param_initializer=tf.initializers.glorot_uniform(), seed=self.seed)
        h_T = get_last_right_output(rnn_encoder, self.max_seq_length, bseq_length, self.rnn_units)
        return h_T


    def compute_rloss(self, logits, y):
        sigmoid_logits = tf.nn.sigmoid(logits)

        neg_y = (1.0 - y)
        pos_logits = y * logits

        pos_max = tf.reduce_max(pos_logits, axis=1)
        pos_max = tf.expand_dims(pos_max, axis=-1)

        pos_min = tf.reduce_min(pos_logits + neg_y * pos_max, axis=1)
        pos_min = tf.expand_dims(pos_min, axis=-1)

        nb_pos, nb_neg = tf.count_nonzero(y, axis=1), tf.count_nonzero(neg_y, axis=1)
        ratio = tf.cast(nb_neg, dtype=tf.float32) / tf.cast(nb_pos, dtype=tf.float32)

        pos_weight = tf.expand_dims(ratio, axis=-1)
        loss = y * -tf.log(sigmoid_logits+1e-8) *pos_weight + neg_y * -tf.log(1.0 - tf.nn.sigmoid(logits - pos_min+1e-8))
        return tf.reduce_mean(loss + 1e-8)

    def compute_closs(self,aug1_h_T,aug2_h_T,h_T):
        bs=int(aug1_h_T.shape[0])
        N=2*bs
        aug_h_T=tf.concat((aug1_h_T,aug2_h_T),axis=0)
        sim=tf.matmul(aug_h_T,tf.transpose(aug_h_T))+1e-8
        sim_i_j=tf.diag_part(tf.slice(sim,[bs,0],[bs,bs]))
        sim_j_i=tf.diag_part(tf.slice(sim,[0,bs],[bs,bs]))
        positive_samples=tf.reshape(tf.concat((sim_i_j,sim_j_i),axis=0),shape=[N,1])
        mask=np.ones((N,N))
        np.fill_diagonal(mask,0)
        for i in range(bs):
            mask[i, bs+i]=0
            mask[bs+i, i]=0
        mask=tf.convert_to_tensor(mask,dtype=tf.bool)
        negative_samples=tf.reshape(tf.boolean_mask(sim,mask),shape=[N,-1])
        labels=tf.zeros(N,tf.int32)
        logits=tf.concat((positive_samples,negative_samples),axis=1)
        closs1=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
        
        aug_h_T=tf.concat((aug1_h_T,h_T),axis=0)
        sim=tf.matmul(aug_h_T,tf.transpose(aug_h_T))
        sim_i_j=tf.diag_part(tf.slice(sim,[bs,0],[bs,bs]))
        sim_j_i=tf.diag_part(tf.slice(sim,[0,bs],[bs,bs]))
        positive_samples=tf.reshape(tf.concat((sim_i_j,sim_j_i),axis=0),shape=[N,1])
        mask=np.ones((N,N))
        np.fill_diagonal(mask,0)
        for i in range(bs):
            mask[i, bs+i]=0
            mask[bs+i, i]=0
        mask=tf.convert_to_tensor(mask,dtype=tf.bool)
        negative_samples=tf.reshape(tf.boolean_mask(sim,mask),shape=[N,-1])
        labels=tf.zeros(N,tf.int32)
        logits=tf.concat((positive_samples,negative_samples),axis=1)
        closs2=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
        
        return closs1+closs2

    
    def compute_quality_at_topk(self, k=10):
        top_k_preds = self.get_topk_tensor(self.predictions, k)
        correct_preds = tf.count_nonzero(tf.multiply(self.y, top_k_preds), axis=1)
        actual_bsize = tf.count_nonzero(self.y, axis=1)
        preds=tf.count_nonzero(top_k_preds,axis=1)
        recall=tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32)
        precision=tf.cast(correct_preds, dtype=tf.float32) / tf.cast(preds, dtype=tf.float32)
        f1_score=(2*recall*precision)/(recall+precision)
        f1_score=tf.where(tf.is_nan(f1_score), tf.zeros(f1_score.shape), f1_score)
        return correct_preds,actual_bsize,f1_score
    
    
    def get_topk_tensor(self, x, k=10):
        _, index_cols = tf.nn.top_k(x, k)

        index_rows = tf.ones(shape=(self.batch_size, k), dtype=tf.int32) * tf.expand_dims(tf.range(0, self.batch_size), axis=-1)

        index_rows = tf.cast(tf.reshape(index_rows, shape=[-1]), dtype=tf.int64)
        index_cols = tf.cast(tf.reshape(index_cols, shape=[-1]), dtype=tf.int64)

        top_k_indices = tf.stack([index_rows, index_cols], axis=1)
        top_k_values = tf.ones(shape=[self.batch_size * k], dtype=tf.float32)

        sparse_tensor = tf.SparseTensor(indices=top_k_indices, values=top_k_values, dense_shape=[self.batch_size, self.nb_items])
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(sparse_tensor))
    
    def relu_with_threshold(self, x, threshold):
        return tf.nn.relu(x - tf.abs(threshold))