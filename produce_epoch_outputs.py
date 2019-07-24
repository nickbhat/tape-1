from typing import Dict, List

import numpy as np 
import os 
import pickle as pkl
import tensorflow as tf

from tape.models.Transformer import Transformer
from tape.data_utils import deserialize_abc_tran_sequence, PFAM_VOCAB

INT_TO_AA = {v: k for k, v in PFAM_VOCAB.items()}

def collect_metrics(path_to_weights: str,
                    path_to_data: str,
                    model: str,
                    lm_type: str,
                    n_layers: int=12,
                    n_heads: int=8,
                    d_model: int=512) -> Dict[str, Dict[str, List[float]]]:
    """
    Returns a set of statistics for each sequence in desired validation set.

    Args:
        path_to_weights: Location of desired weights to load
        path_to_data: Location of desired valid data
        model: One of Transformer, LSTM, or ResNet
        LM_type: How the language model was trained

    Returns:
        metrics: Dict of validation sequence -> results, where results is a dict of -
            encoder_output - Output of LM encoder. Shape [length, d_model]
            single_acc - Accuracy of infilling each individual position given maximal
                         allowed context based on language model. Shape [length, 1]
            single_perplexity - Perplixity of infilling each individual position given
                         maximal allowed context based on language model. Shape [length, 1]
            normal_loss - Loss computed for just this example in standard training setup. Shape [1].
    """
    if model == 'transformer':
        model = Transformer(len(PFAM_VOCAB),
                            n_layers=n_layers, 
                            n_heads=n_heads, 
                            d_model=d_model,
                            d_filter=4*d_model)
    elif model == 'lstm':
        raise NotImplementedError
    elif model == 'resnet':
        raise NotImplementedError

    data = tf.data.TFRecordDataset(path_to_data)
    data = data.map(deserialize_abc_tran_sequence).batch(1)
    iterator = data.make_one_shot_iterator()
    example = iterator.get_next()

    encoder_output_op = model(example)['encoder_output']

    metrics = {}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_weights(path_to_weights)

        ex, encoder_outputs = sess.run([example, encoder_output_op])
        sequence = np.squeeze(ex['primary'])
        sequence = ''.join([INT_TO_AA[i] for i in sequence])

        metrics[sequence] = {}
        metrics[sequence]['encoder_output'] = np.squeeze(encoder_outputs)

    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Produce per epoch results')
    parser.add_argument('epochs', type=lambda s: [int(epoch) for epoch in s.split(',')], help='Choice of epochs to load from')
    parser.add_argument('result_folder', help='Location of model weights')
    parser.add_argument('valid_data_path', help='Location of desired valid data')
    parser.add_argument('save_file', help='Location to save results')
    parser.add_argument('model', choices=['lstm', 'transformer', 'resnet'], help='One of lstm, transformer, or resnet')
    parser.add_argument('lm_type', choices=['masked', 'autoregressive'], help='Method of training language model')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of embedding')
    x = parser.parse_args()

    for epoch in x.epochs:
        path_to_weights = os.path.join(x.result_folder, 'epoch_' + str(epoch) + '.h5')
        metrics = collect_metrics(path_to_weights, x.valid_data_path, x.model, x.lm_type, x.n_layers, x.n_heads, x.d_model)
        save_file = x.save_file
        with open(save_file + '_' + str(epoch) + '.pkl', 'wb') as f:
            pkl.dump(metrics, f)
