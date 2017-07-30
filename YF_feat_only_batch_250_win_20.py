
# coding: utf-8

# In[1]:

import os
import cPickle as pickle
import numpy as np
from nn1_decouple_feat import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print "start"


# In[2]:

from YellowFin_Pytorch.tuner_utils.yellowfin import YFOptimizer


# In[3]:

os.environ['CUDA_VISIBLE_DEVICES']="1"
batch_size = 50
num_classes = 1


# In[4]:

data = pickle.load(open("yf_data.dat", "rb"))
X_train = data['X_train']
X_train_features = data['X_train_features']
Y_marginals = data['Y_marginals']

X_test = data['X_test']
X_test_feature = data['X_test_feature']


# In[5]:

print X_train_features.shape[1], X_test_feature.shape[0]

#test = X_train_features[1:14001:14000/250, :]
test = X_train_features
print "check", test.shape, X_train_features.shape
print "sparsity outside", float(test.size) / float(test.shape[0] * test.shape[1])
print "overall sparsity ", np.sum(np.sum(test, axis=0)!=0)/float(10344)
raw_input()
# In[6]:

word_attn = AttentionWordRNN(batch_size=batch_size, num_tokens=5664, embed_size=100, word_gru_hidden=50, bidirectional= True)
mix_softmax = MixtureSoftmax(batch_size=batch_size, word_gru_hidden = 50, feature_dim = X_train_features.shape[1], n_classes=num_classes)
# mix_softmax = MixtureSoftmax(batch_size=batch_size, word_gru_hidden = 50, feature_dim = 0, n_classes=num_classes)


# In[7]:

softmax = nn.Softmax()
sigmoid = nn.Sigmoid()


# In[8]:

learning_rate = 0.1

optimizer = YFOptimizer(mix_softmax.parameters(), beta=0.999, lr=learning_rate, mu=0.9, zero_debias=True, clip_thresh=None, auto_clip_fac=None, curv_win_width=20)

# word_optmizer = YFOptimizer(word_attn.parameters(), lr=learning_rate, mu=0.0, auto_clip_fac=2.0)
# mix_optimizer = YFOptimizer(mix_softmax.parameters(), lr=learning_rate, mu=0.0, auto_clip_fac=2.0)

criterion = nn.MultiLabelSoftMarginLoss(size_average=True)


# In[9]:

word_attn.cuda()
mix_softmax.cuda()


# In[10]:

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# In[11]:

def plot_func(log_dir, iter_id, loss_list, local_curv_list, max_curv_list, min_curv_list,
             lr_g_norm_list, lr_list, dr_list, mu_list, grad_avg_norm_list,
             dist_list, grad_var_list):
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 
#    plt.figure()
#    plt.semilogy(loss_list, '.', alpha=0.2, label="Loss")
#    plt.semilogy(running_mean(loss_list,100), label="Average Loss")
#    plt.xlabel('Iterations')
#    plt.ylabel('Loss')
#    plt.legend()
#    plt.grid()
#    ax = plt.subplot(111)
#    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
#    plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".pdf")
#    plt.close()
#
#    plt.figure()
#    plt.semilogy(local_curv_list, label="local curvature")
#    plt.semilogy(max_curv_list, label="max curv in win")
#    plt.semilogy(min_curv_list, label="min curv in win")
##         plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
#    plt.semilogy(lr_g_norm_list, label="lr * grad norm")
#    plt.title("On local curvature")
#    plt.grid()
#    ax = plt.subplot(111)
#    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=2, fancybox=True, shadow=True)
#    plt.savefig(log_dir + "/fig_curv_iter_" + str(iter_id) + ".pdf")
#    plt.close()
#
#    plt.figure()
#    plt.semilogy(lr_list, label="lr min")
#    plt.semilogy(dr_list, label="dynamic range")
#    plt.semilogy(mu_list, label="mu")
#    plt.semilogy(grad_avg_norm_list, label="Grad avg norm")
#    plt.semilogy(dist_list, label="Est dist from opt")
#    plt.semilogy(grad_var_list, label="Grad variance")
#    plt.title('LR='+str(lr_list[-1])+' mu='+str(mu_list[-1] ) )
#    plt.grid()
#    plt.legend(loc="upper right")
#    plt.savefig(log_dir + "/fig_hyper_iter_" + str(iter_id) + ".pdf")
#    plt.close()


# In[ ]:

def train_early_stopping(mini_batch_size, X_train, X_train_feature, y_train, X_test, X_test_feature, word_attn_model, sent_attn_model, 
                         optimizer, loss_criterion, num_epoch, 
                         print_val_loss_every = 1000, print_loss_every = 50, print_figure_every=2500):
    start = time.time()
    loss_full = []
    loss_epoch = []
    accuracy_epoch = []
    loss_smooth = []
    accuracy_full = []
    epoch_counter = 0
    print "mini_batch_size", mini_batch_size
    g = gen_minibatch(X_train, X_train_feature, y_train,  mini_batch_size)
    
    # DEBUG
    loss_list = []
    h_max_list = []
    h_min_list = []
    h_list = []
    dist_list = []
    grad_var_list = []
    
    lr_g_norm_list = []
    
    lr_list = []
    dr_list = []
    mu_list = []
    grad_avg_norm_list = []
    
    plot_figure = plt.figure()
    # END of DEBUG
    
    for i in xrange(1, num_epoch + 1):
#         print 'epoch ', i, timeSince(start)

        # DEBUG
#         if i < 2000:
#             optimizer._lr = 0.1
#         optimizer._mu = 0.98
        # END of DEBUG

        try:
            tokens, features, labels = next(g)

#             print labels
#             print 'tokens', tokens
            loss = train_data(tokens, features, labels, word_attn_model, sent_attn_model, optimizer, loss_criterion)
#             print loss
            acc = test_accuracy_mini_batch(tokens, features, labels, word_attn_model, sent_attn_model)
            accuracy_full.append(acc)
            accuracy_epoch.append(acc)
            loss_full.append(loss)
            loss_epoch.append(loss)
            
#             print "optimizer", optimizer._lr, optimizer._mu, optimizer._h_min, optimizer._h_max, optimizer._dist_to_opt, optimizer._grad_var, loss
#             print loss every n passes
            if i % print_loss_every == 0:
                print 'Loss at %d minibatches, %d epoch,(%s) is %f' %(i, epoch_counter, timeSince(start), np.mean(loss_epoch))
                print 'Accuracy at %d minibatches is %f' % (i, np.mean(accuracy_epoch))
        except StopIteration:
            epoch_counter += 1
            print 'Reached %d epocs' % epoch_counter
            print 'i %d' % i
            print 'loss_epoch', np.sum(loss_epoch) / X_train.shape[0]
            print "optimizer", optimizer._lr, optimizer._mu, optimizer._h_min, optimizer._h_max, optimizer._dist_to_opt, optimizer._grad_var

#             print "word_optimizer", word_optmizer._lr, word_optmizer._mu, word_optmizer._h_min, word_optmizer._h_max, word_optmizer._dist_to_opt, word_optmizer._grad_var
#             print "mix_optimizer", mix_optimizer._lr, mix_optimizer._mu, mix_optimizer._h_min, mix_optimizer._h_max, mix_optimizer._dist_to_opt, mix_optimizer._grad_var
#             print loss_epoch
            p = test_accuracy_full_batch(X_test, X_test_feature, mini_batch_size, word_attn_model, sent_attn_model)
#             print p
#             print len(p)
#             p = np.ravel(p)
#             pos = []
#             for i, candidate in enumerate(test_candidates):
#                 test_label_index = L_test.get_row_index(candidate)
#                 test_label       = L_test[test_label_index, 0]
#                 if i < len(p) and p[i] > 0.5:
#                     pos.append(candidate)
                
#             (TP, FP, FN) = entity_level_f1(pos, gold_file, ATTRIBUTE, corpus, parts_by_doc_test)
                
            g = gen_minibatch(X_train, X_train_feature, y_train, mini_batch_size)
            loss_epoch = []
            accuracy_epoch = []
            
        # DEBUG
        loss_list.append(loss)
        h_max_list.append(optimizer._h_max)
        h_min_list.append(optimizer._h_min)
        h_list.append(optimizer._global_state['grad_norm_squared'] )
        
#         print "test in the middle ", h_list[-1], h_max_list[-1], h_min_list[-1]
#         print "curv window ", optimizer._global_state["curv_win"]
        
        dist_list.append(optimizer._dist_to_opt)
        grad_var_list.append(optimizer._grad_var)

        lr_g_norm_list.append(optimizer._lr * np.sqrt(optimizer._global_state['grad_norm_squared'] ) )

        lr_list.append(optimizer._lr)
        dr_list.append(optimizer._h_max / optimizer._h_min)
        mu_list.append(optimizer._mu)
        grad_avg_norm_list = []
#         if (i % 1000 == 0) and i != 0:
#             with open("h_val.txt", "w") as f:
#                 np.savetxt(f, h_list)
        
        if (i % print_figure_every == 0 and i != 0) or (i == 50 or i == 1000):
            plot_func(log_dir=log_dir, iter_id=i, loss_list=loss_list, 
                 local_curv_list=h_list, max_curv_list=h_max_list, 
                 min_curv_list=h_min_list, lr_g_norm_list=lr_g_norm_list, 
                 lr_list=lr_list, dr_list=dr_list, mu_list=mu_list, 
                 grad_avg_norm_list=grad_avg_norm_list,
                 dist_list=dist_list, grad_var_list=grad_var_list)
            print "figure plotted"
        # END of DEBUG
        
#     torch.save(word_attn_model, log_dir + "/word_attn.model")
#     torch.save(sent_attn_model, log_dir + "/sent_attn.model")
            
    return loss_full


# In[ ]:

log_dir = "./YF_sparse_feature_verification_batch_250_new_dist"
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
loss_full= train_early_stopping(batch_size, X_train, X_train_features, Y_marginals, X_test, X_test_feature, word_attn, mix_softmax, optimizer, 
                            criterion, 100000, 1000, 1000)


# In[ ]:




# In[ ]:



