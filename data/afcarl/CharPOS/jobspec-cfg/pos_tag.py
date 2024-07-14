import numpy as np
import theano
import theano.tensor as T
import theano.printing as printing
import lasagne
import cPickle
import lasagne.layers.helper as helper
import random
import argparse
import sys

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

vocab_size = 84 #3

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 256

# Optimization learning rate
LEARNING_RATE = 0.1

# All gradients above this will be clipped
GRAD_CLIP = False

# Number of tags
NUM_TAGS = 46 #2

char_dims = 10
dim_out = 128

def get_mask(x):
    mask = np.ones_like(x)
    mask[np.where(x==0.)] = 0
    return mask 

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main(num_epochs=10, layers=1, load_file=None, batch_size=128, seq_len=96, suffix='', test=False, model_name='model'):
    print "Building network ..."
    print theano.config.floatX

    BATCH_SIZE = batch_size
    SEQ_LENGTH = seq_len

    # Recurrent layers expect input of shape (batch size, SEQ_LENGTH, num_features)
    x = T.imatrix('x')
    mask = T.matrix('mask')
    target_values = T.ivector('target_output')
    
    # We now build a layer for the embeddings.
    U = np.random.randn(vocab_size, char_dims).astype(theano.config.floatX)
    embeddings = theano.shared(U, name='embeddings', borrow=True)
    x_embedded = embeddings[x]

    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, SEQ_LENGTH, char_dims), input_var=x_embedded)
    l_mask = lasagne.layers.InputLayer(shape=(BATCH_SIZE, SEQ_LENGTH), input_var=mask)

    recurrent_type = lasagne.layers.LSTMLayer
    l_forward_1 = recurrent_type(l_in, N_HIDDEN, 
        grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input=l_mask)
    l_backward_1 = recurrent_type(l_in, N_HIDDEN, 
        grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        backwards=True, 
        mask_input=l_mask)
    if layers == 2:
        l_forward_2 = recurrent_type(l_forward_1, N_HIDDEN, 
            grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh,
            mask_input=l_mask)
        l_backward_2 = recurrent_type(l_backward_1, N_HIDDEN, 
            grad_clipping=GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh,
            backwards=True, 
            mask_input=l_mask)
        l_forward_slice = lasagne.layers.get_output(l_forward_2)[:,-1,:]
        l_backward_slice = lasagne.layers.get_output(l_backward_2)[:,-1,:]
    else:
        l_forward_slice = lasagne.layers.get_output(l_forward_1)[:,-1,:]
        l_backward_slice = lasagne.layers.get_output(l_backward_1)[:,-1,:]

    # Now combine the LSTM layers.  
    _Wf, _Wb = np.random.randn(N_HIDDEN, dim_out).astype(theano.config.floatX), np.random.randn(N_HIDDEN, dim_out).astype(theano.config.floatX)
    _bias = np.random.randn(dim_out).astype(theano.config.floatX)
    wf = theano.shared(_Wf, name='join forward weights', borrow=True)
    wb = theano.shared(_Wb, name='join backward weights', borrow=True)
    bias = theano.shared(_bias, name='join bias', borrow=True)

    joined = T.dot(l_forward_slice, wf) + T.dot(l_backward_slice, wb) + bias
    tmp = lasagne.layers.InputLayer(shape=(BATCH_SIZE, dim_out))
    l_out = lasagne.layers.DenseLayer(tmp, num_units=NUM_TAGS, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = l_out.get_output_for(joined)
    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    if layers == 1:
        all_params = helper.get_all_params(l_forward_1) + helper.get_all_params(l_backward_1) 
    else:
        all_params = helper.get_all_params(l_forward_2) + helper.get_all_params(l_backward_2) 
    all_params += helper.get_all_params(l_out) + [wf, wb, bias, embeddings] 
    print len(all_params)

    grads = T.grad(cost, all_params)
    get_grads = theano.function([x, mask, target_values], grads)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adam(cost, all_params)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([x, mask, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([x, mask, target_values], cost, allow_input_downcast=True) 
    
    pred = T.argmax(network_output, axis=1)
    get_preds = theano.function([x, mask], pred, allow_input_downcast=True)

    errors = T.sum(T.neq(pred, target_values))
    count_errors = theano.function([x, mask, target_values], errors, allow_input_downcast=True)

    def get_data(fname):
        import cPickle
        with open(fname, 'rb') as handle:
            data = cPickle.load(handle)
        xs = [d.astype('int32') for d in data[0]]
        return xs, data[1]

    print 'Loading train'
    train_xs, train_ys = get_data('train%s' % suffix)
    print 'Loading dev'
    dev_xs, dev_ys = get_data('dev%s' % suffix)
    print 'Loading test'
    test_xs, test_ys = get_data('test%s' % suffix)
    print 'Sizes:\tTrain: %d\tDev: %d\tTest: %d\n' % (len(train_xs) * BATCH_SIZE, len(dev_xs) * BATCH_SIZE, len(test_xs) * BATCH_SIZE)


    def get_accuracy(pXs, pYs):
        total = sum([len(batch) for batch in pXs])
        errors = sum([count_errors(tx, get_mask(tx), ty) for tx, ty in zip(pXs, pYs)])
        return float(total-errors)/total

    def save_preds(pXs, pYs):
        preds = [get_preds(tx, get_mask(tx)) for tx, _ in zip(pXs, pYs)]
        with open('pred.pkl', 'wb') as handle:
            handle.dump(preds, handle)

    if not load_file is None:
        print 'Loading params...'
        with open(load_file, 'rb') as handle:
            params = cPickle.load(handle)
        print len(params)
        for ix, _ in enumerate(zip(params, all_params)):
            all_params[ix].set_value(params[ix].astype('float32'))

    print("Training ...")
    try:
        if test:
            dev_acc = get_accuracy(dev_xs, dev_ys)
            save_preds(dev_xs, dev_ys)
            print dev_acc
            return

        best_acc = 0.0
        for it in xrange(num_epochs):
            data = zip(train_xs, train_ys)
            random.shuffle(data)
            train_xs, train_ys = zip(*data)

            avg_cost = 0;
            total = 0.
            for x, y in zip(train_xs, train_ys):          
                avg_cost += train(x, get_mask(x), y)
                total += 1.

            train_acc = 0.
            #train_acc = get_accuracy(train_xs, train_ys)
            dev_acc = get_accuracy(dev_xs, dev_ys)
            test_acc = get_accuracy(test_xs, test_ys)

            if dev_acc > best_acc:
                params = [np.asarray(p.eval()) for p in all_params]
                with open('%s_%f.pkl' % (model_name, dev_acc), 'wb') as handle:
                    cPickle.dump(params, handle)
                best_acc = dev_acc

            print("Epoch {} average loss = {}".format(it, avg_cost / total))
            print "Accuracies:\t train: %f\tdev: %f\ttest: %f\n" % (train_acc, dev_acc, test_acc) 
            print 
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--n_recurrent_layers', type=int, default=1, help='How many recurrent layers to stack.')
    parser.add_argument('--load_file', type=str, default=None, help='File to load parameters from.')
    parser.add_argument('--test', type='bool', default=False, help='Whether or not to just test the code.')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for the data files to load.')
    parser.add_argument('--seq_len', type=int, help='Length of the longest sequence.')
    parser.add_argument('--batch_size', type=int, help='Size of the batches in the dataset.')
    parser.add_argument('--model_name', type=str, help='Name of which to save the model.')
    parser.add_argument('--n_epochs', type=int, default=10, help='How long to run training.')
    args = parser.parse_args()
    print "args: ", args
    f = sys.argv[1]
    main(num_epochs=args.n_epochs, 
        layers=args.n_recurrent_layers, 
        load_file=args.load_file, 
        suffix=args.suffix,
        test=args.test,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_name=args.model_name)
