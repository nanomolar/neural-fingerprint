# Example regression script using neural fingerprints.
#
# Compares Morgan fingerprints to neural fingerprints.

import cPickle as pickle
import gflags as flags
import gzip
import logging
import os
import pandas as pd
from sklearn import metrics
import sys
import time

import autograd.numpy as np
import autograd.numpy.random as npr

from neuralfingerprint import adam
from neuralfingerprint import build_batched_grad
from neuralfingerprint import build_conv_deep_net
from neuralfingerprint import build_morgan_deep_net
from neuralfingerprint.build_vanilla_net import binary_classification_nll
from neuralfingerprint.util import rmse

from autograd import grad

flags.DEFINE_string('root', None, 'Root directory for model input.')
flags.DEFINE_string('dataset', None, 'Dataset prefix.')
flags.DEFINE_integer('fold', None, 'Fold index.')
flags.DEFINE_boolean('morgan', False, 'If True, run Morgan experiment.')
FLAGS = flags.FLAGS

logging.getLogger().setLevel(logging.INFO)


def load_cv_data(pattern, fold, num_folds=5):
    """Load cross-validation training and test data.

    The fold index specifies the test fold; the remaining folds are used for
    training.

    Args:
        pattern: String file pattern containing %d for fold index substitution.
        fold: Integer fold index.
        num_folds: Integer number of folds.

    Returns:
        train, valid, test: Tuple of numpy arrays containing SMILES and labels.
    """
    logging.info('Pattern: %s', pattern)
    logging.info('Fold: %d', fold)
    train = []
    valid = []
    test = []
    for fold_idx in range(num_folds):
        with gzip.open(pattern % fold_idx) as f:
            df = pickle.load(f)
        if fold_idx == fold:
            logging.info('Fold %d valid: %d', fold, fold_idx)
            valid.append(df)
        elif fold_idx == (fold + 1) % num_folds:
            logging.info('Fold %d test: %d', fold, fold_idx)
            test.append(df)
        else:
            logging.info('Fold %d train: %d', fold, fold_idx)
            train.append(df)
    train = pd.concat(train)
    valid = pd.concat(valid)
    test = pd.concat(test)

    logging.info('Train: %d\tValid: %d\tTest: %d',
                 len(train), len(valid), len(test))

    train_data = (train['smiles'].values, train['label'].values)
    valid_data = (valid['smiles'].values, valid['label'].values)
    test_data = (test['smiles'].values, test['label'].values)

    return train_data, valid_data, test_data


def xent(predictions, targets):
    """Cross-entropy loss function.

    The arguments are reversed in sklearn.
    """
    # TODO: it might be important to implement this with autograd.numpy as in
    # build_vanilla_net.py.
    try:
        predictions = predictions.value
    except AttributeError:
        pass
    return metrics.log_loss(targets, predictions)

# Usually neural fps need far fewer dimensions than morgan.
# The depth of the network equals the fingerprint radius.
# Only the neural fps need this parameter (conv_width).
# Size of hidden layer of network on top of fps.
model_params = dict(fp_length=50,
                    fp_depth=4,
                    conv_width=20,
                    h1_size=100,
                    L2_reg=np.exp(-2))
train_params = dict(num_iters=10000,
                    batch_size=100,
                    init_scale=np.exp(-4),
                    step_size=np.exp(-6),
                    b1=np.exp(-3),   # Parameter for Adam optimizer.
                    b2=np.exp(-2))   # Parameter for Adam optimizer.

# Define the architecture of the network that sits on top of the fingerprints.
# One hidden layer.
vanilla_net_params = dict(
    layer_sizes=[model_params['fp_length'], model_params['h1_size']],
    normalize=True, L2_reg=model_params['L2_reg'],
    nll_func=binary_classification_nll)


def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_targets,
             train_params, seed=0, validation_smiles=None,
             validation_targets=None, test_smiles=None, test_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    print "Total number of weights in the network:", num_weights
    init_weights = npr.RandomState(seed).randn(num_weights)
    init_weights *= train_params['init_scale']

    step_times = []

    def callback(weights, step):
        step_times.append(time.time())
        if step % 10 == 0:
            logging.info('Step %d', step)
            # Calculate moving average of stepation time.
            logging.info('Average step time: %g',
                         np.mean(np.diff(step_times)[-5:]))
        if step % 100 == 0:
            logging.info('Running valid and test eval')
            # Calculate eval and test metrics.
            start = time.time()
            valid_pred = pred_fun(weights, validation_smiles)
            test_pred = pred_fun(weights, test_smiles)
            logging.info('Eval took %.0fs', time.time() - start)
            validation_auc = metrics.roc_auc_score(validation_targets,
                                                   valid_pred)
            test_auc = metrics.roc_auc_score(test_targets, test_pred)
            logging.info('Validation AUC: %g', validation_auc)
            # Write checkpoint and eval pickles.
            with gzip.open('model.ckpt-%d' % step, 'wb') as f:
                pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
            assert np.array_equal(np.unique(validation_targets), [0, 1])
            assert np.array_equal(np.unique(test_targets), [0, 1])
            rows = [{'dataset': FLAGS.dataset, 'auc': validation_auc,
                     'step': step,
                     'count_0.0': np.count_nonzero(validation_targets == 0),
                     'count_1.0': np.count_nonzero(validation_targets == 1)}]
            with gzip.open('eval-tune-%d.pkl.gz' % step, 'wb') as f:
                pickle.dump(pd.DataFrame(rows), f, pickle.HIGHEST_PROTOCOL)
            rows = [{'dataset': FLAGS.dataset, 'auc': test_auc, 'step': step,
                     'count_0.0': np.count_nonzero(test_targets == 0),
                     'count_1.0': np.count_nonzero(test_targets == 1)}]
            with gzip.open('eval-test-%d.pkl.gz' % step, 'wb') as f:
                pickle.dump(pd.DataFrame(rows), f, pickle.HIGHEST_PROTOCOL)

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun,
                                            train_params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'],
                           step_size=train_params['step_size'],
                           b1=train_params['b1'], b2=train_params['b2'])


def main():
    print "Loading data..."
    pattern = os.path.join(FLAGS.root, FLAGS.dataset + '-fold-%d.pkl.gz')
    traindata, valdata, testdata = load_cv_data(
        pattern, FLAGS.fold, num_folds=5)
    train_inputs, train_targets = traindata
    val_inputs, val_targets = valdata
    test_inputs, test_targets = testdata

    def print_performance(pred_func):
        train_preds = pred_func(train_inputs)
        val_preds = pred_func(val_inputs)
        print "\nPerformance (RMSE) on " + task_params['target_name'] + ":"
        print "Train:", rmse(train_preds, train_targets)
        print "Test: ", rmse(val_preds,  val_targets)
        print "-" * 80
        return rmse(val_preds, val_targets)

    def run_morgan_experiment():
        loss_fun, pred_fun, net_parser = \
            build_morgan_deep_net(model_params['fp_length'],
                                  model_params['fp_depth'], vanilla_net_params)
        num_weights = len(net_parser)
        train_nn(pred_fun, loss_fun, num_weights, train_inputs,
                 train_targets, train_params, validation_smiles=val_inputs,
                 validation_targets=val_targets, test_smiles=test_inputs,
                 test_targets=test_targets)

    def run_conv_experiment():
        conv_layer_sizes = [model_params['conv_width']]
        conv_layer_sizes *= model_params['fp_depth']
        conv_arch_params = {'num_hidden_features': conv_layer_sizes,
                            'fp_length': model_params['fp_length'],
                            'normalize': 1}
        loss_fun, pred_fun, conv_parser = build_conv_deep_net(
            conv_arch_params, vanilla_net_params, model_params['L2_reg'])
        num_weights = len(conv_parser)
        train_nn(pred_fun, loss_fun, num_weights, train_inputs,
                 train_targets, train_params, validation_smiles=val_inputs,
                 validation_targets=val_targets, test_smiles=test_inputs,
                 test_targets=test_targets)

    if FLAGS.morgan:
        logging.info('Starting Morgan fingerprint experiment')
        run_morgan_experiment()
    else:
        logging.info('Starting neural fingerprint experiment')
        run_conv_experiment()

if __name__ == '__main__':
    flags.MarkFlagAsRequired('root')
    flags.MarkFlagAsRequired('dataset')
    flags.MarkFlagAsRequired('fold')
    FLAGS(sys.argv)
    main()
