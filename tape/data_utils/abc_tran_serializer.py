from typing import Dict

from Bio import SeqIO
import random
import string
import tensorflow as tf
import numpy as np
from tqdm import tqdm 
import pickle as pkl

from .vocabs import PFAM_VOCAB

def _bytes_feature(value):
    return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=value)
            )

def _int64_feature(value):
    return tf.train.Feature(
            int64_list=tf.train.Int64List(value=value)
            )

def _float_feature(value):
    return tf.train.Feature(
            float_list=tf.train.FloatList(value=value)
            )

def to_sequence_features(**features):
    for name, array in features.items():
        if array.dtype in [np.int32, np.in64]:
            array = np.asarray(array, np.int32)
            array = [_int64_feature(el) for el in array]
        elif array.dtype in [np.float32, np.float64]:
            array = np.asarray(array, np.float32)
            array = [_float_feature(el) for el in array]
        else: 
            raise TypeError(f'Unrecognized dtype {array.dtype}')
        features[name] = tf.train.FeatureList(feature=array)

    features = tf.train.FeatureLists(feature_list=features)
    return features

def serialize_abc_tran(filename: str, 
                        outfile: str,
                        vocab: Dict[str, int]) -> None:

    with open(filename) as data_file:
        records = list(SeqIO.parse(data_file, 'fasta'))

    random.shuffle(records)

    with tf.io.TFRecordWriter(outfile + '.tfrecord') as writer:
        print('Serializing')
        for record in tqdm(records):
            serialized_entry = serialize_abc_tran_example(record, vocab)
            writer.write(serialized_entry)

def serialize_abc_tran_example(example: SeqIO.SeqRecord,
                                vocab: Dict[str, int]) -> bytes:

    sequence = str(example.seq).replace('.', '')
    name = example.id.split('/')[0].encode('UTF-8')

    int_sequence = [vocab[aa] for aa in list(sequence)]

    protein_context = {}
    protein_context['protein_length'] = _int64_feature([len(sequence)])
    protein_context['pfam_accession'] = _bytes_feature([name])
    protein_context = tf.train.Features(feature=protein_context)

    protein_features = {}
    protein_features['primary'] = [_int64_feature([el]) for el in int_sequence]
    for key, val in protein_features.items():
        protein_features[key] = tf.train.FeatureList(feature=val)
    protein_features = tf.train.FeatureLists(feature_list=protein_features)

    example=tf.train.SequenceExample(context=protein_context, feature_lists=protein_features)
    return example.SerializeToString()

def deserialize_abc_tran_sequence(example) -> Dict[str, tf.Tensor]:
    
    context = {
            'protein_length': tf.io.FixedLenFeature([1], tf.int64),
            'pfam_accession': tf.io.FixedLenFeature([], tf.string)
            }
    
    features = {
            'primary': tf.io.FixedLenSequenceFeature([1], tf.int64)
            }

    context, features = tf.io.parse_single_sequence_example(
            example, 
            context_features=context,
            sequence_features=features
            )

    pfam_accession = context['pfam_accession']
    protein_length = tf.cast(context['protein_length'][0], tf.int32)
    primary = tf.cast(features['primary'][:, 0], tf.int32)

    return {'primary': primary,
            'protein_length': protein_length,
            'pfam_accession': pfam_accession
            }

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser(description='convert swissprot to tfrecords')
    parser.add_argument('filename', type=str, help='filename for .fasta file')
    parser.add_argument('outfile', type=str, help='prefix for .tfrecord file')
    args = parser.parse_args()

    serialize_abc_tran(args.filename, args.outfile, STANDARD_VOCAB)
