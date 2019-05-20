import numpy as np
import theano
import argparse
import time
import sys
import json
import random
import cPickle
import os
import re
from Optimizer import OptimizerList
from Evaluator import Evaluators
from DataManager import DataManager
from TDTC_lstm import AttentionLstm as Model

def train(model, train_data, optimizer, epoch_num, batch_size, batch_n):
    st_time = time.time()
    loss_sum = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    i=0
    for batch in xrange(batch_n):
        print("start batch %d"% i)
        i+=1
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(train_data))
        batch_loss, batch_total_nodes = do_train(model, train_data[start:end], optimizer)
        print('batch_loss[0]:',batch_loss[0],'batch_loss[2]',batch_loss[2])
        loss_sum += batch_loss
        total_nodes += batch_total_nodes

    return loss_sum[0], loss_sum[2]

def do_train(model, train_data, optimizer):
    eps0 = 1e-8
    batch_loss = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    for _, grad in model.grad.iteritems():
        grad.set_value(np.asarray(np.zeros_like(grad.get_value()), \
                dtype=theano.config.floatX))
    for item in train_data:
	sequences_L, sequences_R, target, tar_scalar, solution =  item['seqs_L'], item['seqs_R'], item['target'], item['target_index'], item['solution']
        batch_loss += np.array(model.func_train(sequences_L, sequences_R, tar_scalar, solution))
        total_nodes += len(solution)
    for _, grad in model.grad.iteritems():
        grad.set_value(grad.get_value() / float(len(train_data)))
    optimizer.iterate(model.grad)
    return batch_loss, total_nodes

def test(model, test_data, grained, index_to_word):
    evaluator = Evaluators[grained]()
    keys = evaluator.keys()
    def cross(solution, pred):
        return -np.tensordot(solution, np.log(pred), axes=([0, 1], [0, 1]))

    loss = .0
    total_nodes = 0
    correct = {key: np.array([0]) for key in keys}
    wrong = {key: np.array([0]) for key in keys}
    for item in test_data:
	sequences_L, sequences_R, target, tar_scalar, solution =  item['seqs_L'], item['seqs_R'], item['target'], item['target_index'], item['solution']
        pred = model.func_test(sequences_L, sequences_R, tar_scalar)
        loss += cross(solution, pred)
        total_nodes += len(solution)
        pred_max=np.argmax(pred, axis=1)
        result = evaluator.accumulate(solution[-1:], pred[-1:])
        f1=open('test.out','a')
        sequences_L1 = sequences_L[:-len(target)]
        for index in sequences_L1:                       
	        f1.write(index_to_word[index-1])
		f1.write(' ')
        for index in list(reversed(sequences_R)):
                f1.write(index_to_word[index-1])
                f1.write(' ')
        f1.write('\n')
        for index in target:
        	f1.write(index_to_word[index-1])
        f1.write('\n')
        f1.write('label:')
        f1.write(str(np.argmax(solution, axis=1)[0]-1))
        f1.write('\n')
        f1.write('prediction:')
        f1.write(str(pred_max[0]-1))
        f1.write('\n')
        f1.close()
    acc = evaluator.statistic()
    return loss/total_nodes, acc

def classify(model, data, grained, index_to_word, save_dir):
    evaluator = Evaluators[grained]()
    keys = evaluator.keys()
    correct = {key: np.array([0]) for key in keys}
    wrong = {key: np.array([0]) for key in keys}
    for item in data:
        
		sequences_L, sequences_R, target, tar_scalar, solution =  item['seqs_L'], item['seqs_R'], item['target'], item['target_index'], item['solution']
        pred = model.func_test(sequences_L, sequences_R, tar_scalar)
		
        pred_max=np.argmax(pred, axis=1)
        f1=open(save_dir,'a')
        sequences_L1 = sequences_L[:-len(target)]
        for index in sequences_L1:                       
	        f1.write(index_to_word[index-1])
		f1.write(' ')
        for index in list(reversed(sequences_R)):
		f1.write(index_to_word[index-1])
		f1.write(' ')
        f1.write('\n')
        for index in target:
        	f1.write(index_to_word[index-1])

        f1.write('\n')
        f1.write('prediction:')
        f1.write(str(pred_max[0]-1))
        f1.write('\n')
        f1.close()
    return 0


if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='lstm')
    parser.add_argument('--seed', type=int, default=int(1000*time.time()))
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--dim_gram', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--fast', type=int, choices=[0, 1], default=0)
    parser.add_argument('--screen', type=int, choices=[0, 1], default=0)
    parser.add_argument('--optimizer', type=str, default='ADAGRAD')
    parser.add_argument('--grained', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_word_vector', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch', type=int, default=25)
    parser.add_argument('--is_train', type=bool, default=False)
    args, _ = parser.parse_known_args(argv)
    is_train = args.is_train
    random.seed(args.seed)
    if is_train:   
		data = DataManager(args.dataset)
		wordlist = data.gen_word()
		train_data, test_data, classify_data  = data.gen_data(args.grained)
		model = Model(wordlist, argv, len(data.dict_target))
		batch_n = (len(train_data)-1) / args.batch + 1
		optimizer = OptimizerList[args.optimizer](model.params, args.lr, args.lr_word_vector)
		details = {'loss': [], 'loss_train':[], 'loss_dev':[], 'loss_test':[], \
            'acc_train':[], 'acc_dev':[], 'acc_test':[], 'loss_l2':[]}
		index_to_word=[]
		index_to_word=data.index_to_word
		f=open('word.txt','wb')
		for word in index_to_word:
			f.write(word)
			f.write('\n')
		for e in range(args.epoch):
   
			print('train %d epoch:' %e)
      	    random.shuffle(train_data)
			now = {}
			now['loss'], now['loss_l2'] = train(model, train_data, optimizer, e, args.batch, batch_n)
    
        	print('save model params in model/model_params.txt')
        	f = open('model/model_params.txt','wb')
        	for param in model.params:
        		cPickle.dump(param,f,protocol=cPickle.HIGHEST_PROTOCOL)
        	f.close()
        
        	print('calculate the accuracy and loss on test data')
        	now['loss_test'], now['acc_test'] = test(model, test_data, args.grained, index_to_word)
            print('acc_test:',now['acc_test'])
        	for key, value in now.items(): 
  	        	details[key].append(value)
        	with open('result/%s.txt' % args.name, 'w') as f:
      			f.writelines(json.dumps(details))
    else:
		app = Application()
		save_dir = app.save_file
		
		app.master.title('TD classification')
		app.mainloop()
		data = DataManager(args.dataset)
		wordlist = data.gen_word()
		_, _, classify_data  = data.gen_data(args.grained)
		model = Model(wordlist, argv, len(data.dict_target))
		index_to_word=[]
		index_to_word=data.index_to_word
		
        classify(model, classify_data, args.grained, index_to_word, save_dir)
