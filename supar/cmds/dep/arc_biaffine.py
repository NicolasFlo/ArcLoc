# -*- coding: utf-8 -*-

import argparse

from supar import ArcBiaffineDependencyParser
from supar.cmds.cmd import init


def main():
    parser = argparse.ArgumentParser(description='Create Biaffine Dependency Parser with arc representations.',
                                     allow_abbrev=False)
    parser.set_defaults(Parser=ArcBiaffineDependencyParser)
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivize the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.add_argument('--arc_transformer_choice', default=1, type=int, help='Choice of transformer for arc representations. \n'\
                           ' 0: None\n'\
                           ' 1: Pytorch TransformerEncoder'\
                           ' 2: Linear Transformer\n' \
                           'Default: 1.')
    parser.add_argument('--rel_transformer_choice', default=1, type=int, help='Choice of transformer for rel representations. \n'\
                           ' 0: None\n'\
                           ' 1: Pytorch TransformerEncoder'\
                           ' 2: Linear Transformer\n' \
                           'Default: 1.')
    parser.add_argument('--arc_size', default=512, type=int, help='Size of arc representations. Default: 512.')
    parser.add_argument('--rel_size', default=512, type=int, help='Size of rel representations. Default: 512.')
    parser.add_argument('--n_heads_arcs', default=32, type=int, help="Number of attention heads for arc transformers that have them. Default: 32.")
    parser.add_argument('--n_heads_rels', default=32, type=int, help="Number of attention heads for rel transformers that have them. Default: 32.")
    parser.add_argument('--n_local_heads', default=0, type=int, help='Number of local attention heads for transformers that have them. Default: 0.')
    parser.add_argument('--n_layers_arcs', default=1, type=int, help="Number of layers for arc transformers. Default: 1.")
    parser.add_argument('--n_layers_rels', default=1, type=int, help="Number of layers for rel transformers. Default: 1.")
    parser.add_argument('--filter_dropout_rate', default=.33, type=float, help='The dropout ratio of the filter MLP. Default: .33.')
    parser.add_argument('--scoring_dropout_rate', default=.33, type=float, help='The dropout ratio of the scoring MLPs. Default: .33.')
    parser.add_argument('--transformer_encoder_dropout_rate', default=.33, type=float, help='The dropout ratio of the GateTransformerEncoderLayer. Default: .33.')
    parser.add_argument('--max_seq', default=8192, type=int, help='Max sequence length for relevant transformers. Default: 8192.')
    parser.add_argument('--masked_arc_scorer', default='mlp', help='Choice between a multilayer perceptron or a linear transformation '\
                        'to get the first scores of the arcs to be used to create a filter (mlp or lin). Default: mlp.')
    parser.add_argument('--separate_scoring', default="True", help="If True, the arcs and rels scoring will be done using 2 MLPs "\
                        "(or linear transformations) else only 1 is used and its output is separated into arc scores and rel scores. "\
                        "Default: True.")
    parser.add_argument('--arc_resize_factor', default=1, type=float, help="Factor of increase of the output size of the arcs' Biaffine module "\
                        "(arc_size*arc_resize_factor) The vectors will then be projected to arc_size using a linear transformation.\n"\
                        "Since the final arc size is int(arc_size*arc_resize_factor) it's important for the factor have a sensible "\
                        "value to either increase or decrease the arc size without going below 1 or above number so big the arcs "\
                        "won't fit in the memory. Default: 1.")
    parser.add_argument('--rel_resize_factor', default=1, type=float, help="Factor of increase of the output size of the rels' Biaffine module "\
                        "(rel_size*rel_resize_factor) The vectors will then be projected to rel_size using a linear transformation.\n"\
                        "Since the final rel size is int(rel_size*rel_resize_factor) it's important for the factor have a sensible "\
                        "value to either increase or decrease the rel size without going below 1 or above number so big the rels "\
                        "won't fit in the memory. Default: 1.")
    parser.add_argument('--use_biaffine_filter', action='store_true',
                        help='Whether or not to filter the arcs in the Biaffine module and only compute the best ones.')
    parser.add_argument('--use_layer_filter', action='store_true',
                        help='Whether or not to filter the arcs computed by the Biaffine module to only give the best ones to the transformer.')
    parser.add_argument('--learn_filter', action='store_true',
                        help='Whether or not to learn the arc filter (True) or just use the arc scores to choose the best arcs.')
    parser.add_argument('--filter_factor', default=3, type=int,
                        help="Choose how many of the best arcs will be chosen if a filter is used.\n"\
                        'Default: 3')
    parser.add_argument('--use_gumbel_noise', default="True", help="Whether or not to add noise from a gumbel distribution to the initial filter scores.\n"\
                        "Default: True.")
    parser.add_argument('--use_gumbel_sampling', default="True", help="Whether to sample the k best head candidates when filtering using a gumbel softmax or a simple sort (False).\n"\
                        "Default: True.")
    parser.add_argument('--gumbel_scale', default=0.005, type=float,
                        help="The scale of the gumbel distribution to be added to the filter scores.\n"\
                        "Default: 0.005.")
    parser.add_argument('--use_layernorm', default="True", help="If True, use layernorms before and after the transformer."\
                        "Default: True.")
    parser.add_argument('--arc_rel_scale', default=0.5, type=float,
                        help="Rescaling of arc_loss and rel_loss in the loss function, between 0 (favoring rel_loss) and 1 (favoring arc_loss).\n"\
                        "arc_loss*=2*arc_rel_scale\n"\
                        "rel_loss*=2*(1-arc_rel_scale)\n"\
                        "Default: 0.5.")
    parser.add_argument('--use_word_embeddings_for_biaffine', action='store_true',
                        help="Use no MLP module for head/dependent specilization, instead sue the word embeddings as the Biaffine's input.")
    parser.add_argument('--use_rel_mlp', action='store_true',
                        help="Use a specific MLP module for the labels for head/dependent specilization (as opposed to sharing the arc's MLPs' outputs).")
    parser.add_argument('--use_rel_biaffine', action='store_true',
                        help="Use a specific Biaffine module to create vector representations of the labels (as opposed to sharing the arc's vector representations).")
    parser.add_argument('--use_rel_transformer', action='store_true',
                        help="Use a specific transformer for label vectors, otherwise arc and label vectors will use the same transformer, requiring arc_size = rel_size.")
    parser.add_argument('--separate_features', default="False",
                        help="If True, uses parts of the Biaffine vector representation for arc scoring and another part for label scoring.\n"\
                        "This requires not using specific rel vectors and to use separate scoring.\n"
                        "Default: False")
    parser.add_argument('--feature_scale', default=0.5, type=float,
                        help="Rescaling factor for arc and label scoring if separate_features is True, "\
                        "then the first arc_size*(1-feature_scale) features are used for arc scoring.\n"\
                        "And the last arc_size*feature_scale are used for rel scoring\n"\
                        "Default: 0.5.")
    parser.add_argument('--no_biaffine', action='store_true',
                        help="Do not use a biaffine module to create arc vectors, instead use an MLP.")
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'elmo', 'bert'], nargs='*', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--encoder', choices=['lstm', 'transformer', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.conllx', help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx', help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx', help='path to test file')
    subparser.add_argument('--embed', default='glove-6b-100', help='file or embeddings available at `supar.utils.Embedding`')
    subparser.add_argument('--bert', default='bert-base-cased', help='which BERT model to use')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'elmo', 'bert'], nargs='*', help='features to use')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--bert', default='bert-base-cased', help='which BERT model to use')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--bert', default='bert-base-cased', help='which BERT model to use')
    init(parser)


if __name__ == "__main__":
    main()
