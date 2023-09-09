import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import sys
import utils
import time
    

def train_network(sess, net, train_generator, validate_generator, nb_epoch, 
                  total_train_batches, total_validate_batches, display_step,
                  early_stopping_k, output_dir,
                  test_generator, total_test_batches, log_dir):
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    f=open(log_dir,'w')
    val_best_performance = sys.float_info.min
    patience_cnt = 0
    for epoch in range(0, nb_epoch):
        f.write("\n=========================================\n")
        f.write("@Epoch#" + str(epoch)+"\n")

        train_loss = 0.0
        train_hits5 =0 
        train_actual5=0
        train_f1_score5 = 0.0

        train_hits10 = 0
        train_actual10=0
        train_f1_score10 = 0.0

        train_hits20 = 0
        train_actual20=0
        train_f1_score20 = 0.0

        train_hits30 = 0
        train_actual30=0
        train_f1_score30 = 0.0
        for batch_id, data in train_generator:
            start_time = time.time()
            loss, correct_preds5,actual_bsize5,f1_score5, correct_preds10,actual_bsize10,f1_score10,correct_preds20,actual_bsize20,f1_score20,correct_preds30,actual_bsize30,f1_score30 = net.train_batch(data['S'], data['L'], data['Y'])

            train_loss += loss
            avg_train_loss = train_loss / (batch_id + 1)/32


            train_hits5 += np.sum(correct_preds5)
            train_actual5+=np.sum(actual_bsize5)
            train_hr5=train_hits5/train_actual5

            train_f1_score5 += np.sum(f1_score5)
            avg_train_f1_score5 = train_f1_score5 / (batch_id + 1)/32


            train_hits10 += np.sum(correct_preds10)
            train_actual10+=np.sum(actual_bsize10)
            train_hr10=train_hits10/train_actual10

            train_f1_score10 += np.sum(f1_score10)
            avg_train_f1_score10 = train_f1_score10 / (batch_id + 1)/32



            train_hits20 += np.sum(correct_preds20)
            train_actual20+=np.sum(actual_bsize20)
            train_hr20=train_hits20/train_actual20

            train_f1_score20 += np.sum(f1_score20)
            avg_train_f1_score20 = train_f1_score20 / (batch_id + 1)/32



            train_hits30 += np.sum(correct_preds30)
            train_actual30+=np.sum(actual_bsize30)
            train_hr30=train_hits30/train_actual30

            train_f1_score30 += np.sum(f1_score30)
            avg_train_f1_score30 = train_f1_score30 / (batch_id + 1)/32
            

            if batch_id % display_step == 0 or batch_id == total_train_batches - 1:
                running_time = time.time() - start_time
                f.write("Training | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_train_batches) 
                    + " | Loss= " + "{:.8f}".format(avg_train_loss)  
                    + " | hr@5 = " + "{:.8f}".format(train_hr5) 
                    + " | f1_score@5 = " + "{:.8f}".format(avg_train_f1_score5)

                    + " | hr@10 = " + "{:.8f}".format(train_hr10) 
                    + " | f1_score@10 = " + "{:.8f}".format(avg_train_f1_score10)
                    
                    + " | hr@20 = " + "{:.8f}".format(train_hr20) 
                    + " | f1_score@20 = " + "{:.8f}".format(avg_train_f1_score20)
                    
                    + " | hr@30 = " + "{:.8f}".format(train_hr30) 
                    + " | f1_score@30 = " + "{:.8f}".format(avg_train_f1_score30)
                    + " | Time={:.2f}".format(running_time) + "s\n")

            if batch_id >= total_train_batches - 1:
                break

        f.write("\n-------------- VALIDATION LOSS--------------------------\n")
        val_loss = 0.0
        val_hits5 =0 
        val_actual5=0
        val_f1_score5 = 0.0

        val_hits10 = 0
        val_actual10=0
        val_f1_score10 = 0.0

        val_hits20 = 0
        val_actual20=0
        val_f1_score20 = 0.0

        val_hits30 = 0
        val_actual30=0
        val_f1_score30 = 0.0
        for batch_id, data in validate_generator:
            loss, correct_preds5,actual_bsize5,f1_score5, correct_preds10,actual_bsize10,f1_score10,correct_preds20,actual_bsize20,f1_score20,correct_preds30,actual_bsize30,f1_score30 = net.validate_batch(data['S'], data['L'], data['Y'])

            val_loss += loss
            avg_val_loss = val_loss / (batch_id + 1)/32

            val_hits5 += np.sum(correct_preds5)
            val_actual5+=np.sum(actual_bsize5)
            val_hr5=val_hits5/val_actual5

            val_f1_score5 += np.sum(f1_score5)
            avg_val_f1_score5 = val_f1_score5 / (batch_id + 1)/32


            val_hits10 += np.sum(correct_preds10)
            val_actual10+=np.sum(actual_bsize10)
            val_hr10=val_hits10/val_actual10

            val_f1_score10 += np.sum(f1_score10)
            avg_val_f1_score10 = val_f1_score10 / (batch_id + 1)/32



            val_hits20 += np.sum(correct_preds20)
            val_actual20+=np.sum(actual_bsize20)
            val_hr20=val_hits20/val_actual20

            val_f1_score20 += np.sum(f1_score20)
            avg_val_f1_score20 = val_f1_score20 / (batch_id + 1)/32



            val_hits30 += np.sum(correct_preds30)
            val_actual30+=np.sum(actual_bsize30)
            val_hr30=val_hits30/val_actual30

            val_f1_score30 += np.sum(f1_score30)
            avg_val_f1_score30 = val_f1_score30 / (batch_id + 1)/32
            

            if batch_id % display_step == 0 or batch_id == total_validate_batches - 1:
                running_time = time.time() - start_time
                f.write("Validate | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_validate_batches) 
                    + " | Loss= " + "{:.8f}".format(avg_val_loss)  
                    + " | hr@5 = " + "{:.8f}".format(val_hr5) 
                    + " | f1_score@5 = " + "{:.8f}".format(avg_val_f1_score5)

                    + " | hr@10 = " + "{:.8f}".format(val_hr10) 
                    + " | f1_score@10 = " + "{:.8f}".format(avg_val_f1_score10)
                    
                    + " | hr@20 = " + "{:.8f}".format(val_hr20) 
                    + " | f1_score@20 = " + "{:.8f}".format(avg_val_f1_score20)
                    
                    + " | hr@30 = " + "{:.8f}".format(val_hr30) 
                    + " | f1_score@30 = " + "{:.8f}".format(avg_val_f1_score30)
                    + " | Time={:.2f}".format(running_time) + "s\n")
            if batch_id >= total_validate_batches - 1:
                break

        avg_val_loss = val_loss / total_validate_batches
        f.write("\n@ The validation's loss = " + str(avg_val_loss)+"\n")
        if val_best_performance < val_f1_score5:
            val_best_performance=val_f1_score5

            patience_cnt = 0

            save_dir = output_dir + "/epoch_" + str(epoch)
            utils.create_folder(save_dir)

            save_path = saver.save(sess, save_dir + "/model.ckpt")
            f.write("The model is saved in: %s" % save_path)

            f.write("\n-------------- TEST LOSS--------------------------\n")
            test_loss = 0.0
            test_hits5 =0 
            test_actual5=0
            test_f1_score5 = 0.0

            test_hits10 = 0
            test_actual10=0
            test_f1_score10 = 0.0

            test_hits20 = 0
            test_actual20=0
            test_f1_score20 = 0.0

            test_hits30 = 0
            test_actual30=0
            test_f1_score30 = 0.0

            for batch_id, data in test_generator:
                loss, correct_preds5,actual_bsize5,f1_score5, correct_preds10,actual_bsize10,f1_score10,correct_preds20,actual_bsize20,f1_score20,correct_preds30,actual_bsize30,f1_score30 = net.validate_batch(data['S'], data['L'], data['Y'])

                test_loss += loss
                avg_test_loss = test_loss / (batch_id + 1)/32

                test_hits5 += np.sum(correct_preds5)
                test_actual5+=np.sum(actual_bsize5)
                test_hr5=test_hits5/test_actual5

                test_f1_score5 += np.sum(f1_score5)
                avg_test_f1_score5 = test_f1_score5 / (batch_id + 1)/32


                test_hits10 += np.sum(correct_preds10)
                test_actual10+=np.sum(actual_bsize10)
                test_hr10=test_hits10/test_actual10

                test_f1_score10 += np.sum(f1_score10)
                avg_test_f1_score10 = test_f1_score10 / (batch_id + 1)/32



                test_hits20 += np.sum(correct_preds20)
                test_actual20+=np.sum(actual_bsize20)
                test_hr20=test_hits20/test_actual20

                test_f1_score20 += np.sum(f1_score20)
                avg_test_f1_score20 = test_f1_score20 / (batch_id + 1)/32



                test_hits30 += np.sum(correct_preds30)
                test_actual30+=np.sum(actual_bsize30)
                test_hr30=test_hits30/test_actual30

                test_f1_score30 += np.sum(f1_score30)
                avg_test_f1_score30 = test_f1_score30 / (batch_id + 1)/32
                
                if batch_id % display_step == 0 or batch_id == total_test_batches - 1:
                    running_time = time.time() - start_time
                    f.write("Test | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_test_batches) 
                        + " | Loss= " + "{:.8f}".format(avg_test_loss)  
                        + " | hr@5 = " + "{:.8f}".format(test_hr5) 
                        + " | f1_score@5 = " + "{:.8f}".format(avg_test_f1_score5)

                        + " | hr@10 = " + "{:.8f}".format(test_hr10) 
                        + " | f1_score@10 = " + "{:.8f}".format(avg_test_f1_score10)
                        
                        + " | hr@20 = " + "{:.8f}".format(test_hr20) 
                        + " | f1_score@20 = " + "{:.8f}".format(avg_test_f1_score20)
                        
                        + " | hr@30 = " + "{:.8f}".format(test_hr30) 
                        + " | f1_score@30 = " + "{:.8f}".format(avg_test_f1_score30)
                        + " | Time={:.2f}".format(running_time) + "s\n")

                if batch_id >= total_test_batches - 1:
                    break
        else:
            patience_cnt += 1
            f.write("#Patience "+str(patience_cnt)+" out of "+str(early_stopping_k))
        if patience_cnt >= early_stopping_k:
            f.write("# The training is early stopped at Epoch " + str(epoch)+"\n")
            break
    f.close()


def recent_model_dir(dir):
    folder_list = utils.list_directory(dir, True)
    folder_list = sorted(folder_list, key=get_epoch)
    return folder_list[-1]


def get_epoch(x):
    idx = x.index('_') + 1
    return int(x[idx:])
