# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import distributions
from torch.distributions import gumbel
import math
from supar.model import Model
from supar.modules import MLP, Biaffine
from supar.structs import DependencyCRF, MatrixTree
from supar.utils import Config
from supar.utils.common import MIN
from supar.utils.transform import CoNLL

from supar.transformers import Kernel_transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, functional


class ArcBiaffineDependencyModel(Model):
    r"""
    Based on the implementation fo Biaffine Dependency Parser :cite:`dozat-etal-2017-biaffine`.
    Biaffine attention is replaced by a transformer that learns from representations of the arcs.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 800.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        no_mlp_specialization (bool):
            If True, the MLP feature extraction will be done with only 1 MLP without regard for the roles.
            Else different MLPs will be used for each role. Default: False.
        use_hidden_layers_in_specialization_mlp (bool):
            If ``True``, a hidden layer will be added to the MLPs which specialize the encoder's vectors into 1 or more
            roles, else no hidden layer will be added and no activation function will be used on the last layer.
            Default: ``False``.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        filter_dropout_rate (float):
            The dropout ratio of the filter MLP. Default: .33.
        scoring_dropout_rate (float):
            The dropout ratio of the scoring MLPs. Default: .33.
        transformer_encoder_dropout_rate (float):
            The dropout ratio of the GateTransformerEncoderLayer. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.
        arc_transformer_choice (int):
            Choice of transformer for arc representations:
                        0: No transformer
                        1: Pytorch TransformerEncoder
                        2: Linear Transformer
                        Default: 1.
        rel_transformer_choice (int):
            Choice of transformer for rel representations:
                        0: No transformer
                        1: Pytorch TransformerEncoder
                        2: Linear Transformer
                        Default: 1.
        arc_size (int):
            Size of arc representations. Default: 512.
        rel_size (int):
            Size of rel representations. Default: 512.
        n_heads_arcs (int):
            Number of attention heads for arcs' transformers that have them. Default: 32.
        n_heads_rels (int):
            Number of attention heads for rels' transformers that have them. Default: 32.
        n_local_heads (int):
            Number of local attention heads for transformers that have them. Default: 0.
        n_layers_arcs (int):
            Number of layers for arc transformers. Default: 1.
        n_layers_rels (int):
            Number of layers for rel transformers. Default: 1.
        max_seq (int):
            Max sequence length for relevant transformers. Default: 8192.
        masked_arc_scorer:
            Choice between a multilayer perceptron or a linear transformation to get the first scores of the arcs to be used to create a filter (mlp or lin). Default: mlp.
        arc_scorer:
            Choice between a multilayer perceptron or a linear transformation to get the scores of the arcs (mlp or lin). Default: mlp.
        rel_scorer:
            Choice between a multilayer perceptron or a linear transformation to get the scores of the rels (mlp or lin). Default: mlp.
        separate_scoring (bool):
            If True, the arcs and rels scoring will be done using 2 MLPs (or linear transformations), else only 1 is used and
            its output is separated into arc scores and rel scores. Default: True.
        layer_norm_choice (str):
            Choice of layer normalization to be used by the linear transformer or cosformer.
            ``'layer'``: LayerNorm
            ``'RMS'``: RMSNorm
            ``'None'``: None
            Default: ``'layer'``.
        arc_resize_factor (float):
            Factor of increase of the output size of the arcs' Biaffine module (arc_size*arc_resize_factor)
            The vectors will then be projected to arc_size using a linear transformation.
            Since the final arc size is int(arc_size*arc_resize_factor) it's important for the
            factor have a sensible value to either increase or decrease the arc size without
            going below 1 or above number so big the arcs won't fit in the memory.
            Default: 1.
        rel_resize_factor (float):
            Factor of increase of the output size of the rels' Biaffine module (rel_size*rel_resize_factor)
            The vectors will then be projected to rel_size using a linear transformation.
            Since the final rel size is int(rel_size*rel_resize_factor) it's important for the
            factor have a sensible value to either increase or decrease the rel size without
            going below 1 or above number so big the rels won't fit in the memory.
            Default: 1.
        use_biaffine_filter (bool):
            Whether or not to filter the arcs in the Biaffine module and only compute the best ones.
            Default: False.
        use_layer_filter (bool):
            Whether or not to filter the arcs computed by the Biaffine module to only give the best ones to the transformer.
            Default: False.
        learn_filter (bool):
            Whether or not to learn the arc filter (True) or just use the arc scores to choose the best arcs.
            Default: False.
        filter_factor (int):
            Choose how many of the best arcs will be chosen if a filter is used. Default: 3.
        use_gumbel_noise (bool):
            Whether or not to add noise from a gumbel distribution to the initial filter scores. Default: True.
        use_gumbel_sampling (bool):
            Whether to sample the k best head candidates when filtering using a gumbel softmax or a simple sort (False). Default: True.
        gumbel_scale (float):
            The scale of the gumbel distribution to be added to the filter scores. Default: 0.005.
        use_layernorm (bool):
            If True, use layernorms before and after the transformer. Default: True.
        arc_rel_scale (float):
            Rescaling of arc_loss and rel_loss in the loss function, between 0 (favoring rel_loss) and 1 (favoring arc_loss).
            arc_loss *= 2*arc_rel_scale
	    rel_loss *= 2*(1-arc_rel_scale)
            Default: 0.5.
        use_word_embeddings_for_biaffine (bool):
            Use no MLP module for head/dependent specilization, instead sue the word embeddings as the Biaffine's input.
        use_rel_mlp (bool):
            Use no MLP module for head/dependent specilization, instead compute arc representation directly from word embeddings.
        use_rel_biaffine (bool):
            Use a specific Biaffine module to create vector representations of the labels (as opposed to sharing the arc's vector representations).
        use_rel_transformer (bool):
            Use a specific transformer for label vectors, otherwise arc and label vectors will use the same transformer, requiring arc_size = rel_size.
        separate_features (bool):
            If True, uses parts of the Biaffine vector representation for arc scoring and another part for label scoring.
            This requires not using specific rel vectors and to use separate scoring.
            Default: False
        feature_scale (float):
            Rescaling factor for arc and label scoring if separate_features is True,
            then the first arc_size*(1-feature_scale) features are used for arc scoring.
            And the last arc_size*feature_scale are used for rel scoring.
            Default: 0.5.
        no_biaffine (bool):
            Do not use a biaffine module to create arc vectors, instead use an MLP.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_rels,
                 no_mlp_specialization,
                 use_hidden_layers_in_specialization_mlp,
                 arc_transformer_choice,
                 rel_transformer_choice,
                 arc_size,
                 rel_size,
                 n_heads_arcs,
                 n_heads_rels,
                 n_local_heads,
                 n_layers_arcs,
                 n_layers_rels,
                 max_seq,
                 masked_arc_scorer,
                 arc_scorer,
                 rel_scorer,
                 separate_scoring,
                 use_biaffine_filter,
                 use_layer_filter,
                 learn_filter,
                 filter_factor,
                 use_gumbel_noise,
                 use_gumbel_sampling,
                 gumbel_scale,
                 use_layernorm,
                 arc_rel_scale,
                 use_word_embeddings_for_biaffine,
                 use_rel_mlp,
                 use_rel_biaffine,
                 use_rel_transformer,
                 separate_features,
                 feature_scale,
                 no_biaffine,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.33,
                 n_encoder_hidden=800,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 filter_dropout_rate=.33,
                 scoring_dropout_rate=.33,
                 transformer_encoder_dropout_rate=0.33,
                 arc_resize_factor=1,
                 rel_resize_factor=1,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        self.n_encoder_hidden = n_encoder_hidden
        self.n_arc_mlp = n_arc_mlp
        self.n_rel_mlp = n_rel_mlp
        self.n_rels = n_rels
        self.no_mlp_specialization = eval(no_mlp_specialization)
        use_hidden_layers_in_specialization_mlp = eval(use_hidden_layers_in_specialization_mlp)
        self.arc_size = arc_size
        self.rel_size = rel_size
        self.arc_resize_factor = arc_resize_factor
        self.rel_resize_factor = rel_resize_factor
        self.arc_transformer_choice = arc_transformer_choice
        self.rel_transformer_choice = rel_transformer_choice
        self.use_causal_mask = use_causal_mask
        self.use_biaffine_filter = use_biaffine_filter
        self.use_layer_filter = use_layer_filter
        self.learn_filter = learn_filter
        self.filter_factor = filter_factor
        self.use_gumbel_noise = eval(use_gumbel_noise)
        self.use_gumbel_sampling = eval(use_gumbel_sampling)
        self.gumbel_scale = gumbel_scale
        self.separate_scoring = eval(separate_scoring)
        self.use_layernorm = eval(use_layernorm)
        self.arc_rel_scale = arc_rel_scale
        self.use_word_embeddings_for_biaffine = use_word_embeddings_for_biaffine
        self.use_rel_mlp = use_rel_mlp
        self.use_rel_biaffine = use_rel_biaffine
        self.use_rel_transformer = use_rel_transformer
        self.separate_features = eval(separate_features)
        self.feature_scale = feature_scale
        self.no_biaffine = no_biaffine
        
        #Decide whether or not to use normalizations before MLPs
        self.use_norm_arc_mlp = False
        self.use_norm_scorer_mlp = False

        if pre_mlp_norm_choice >= 2:
            self.use_norm_scorer_mlp = True
            pre_mlp_norm_choice-=2
        if pre_mlp_norm_choice >= 1:
            self.use_norm_arc_mlp = True

        self.criterion_arc = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.01)
        self.criterion_rel = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.01)
        self.filter_criterion = nn.BCEWithLogitsLoss()

        # If we're directly going to use arc representations as the Biaffine's input, we don't need these MLPs and LayerNorm
        if not self.use_word_embeddings_for_biaffine:
            if self.use_norm_arc_mlp:
                self.mlp_norm = LayerNorm(self.args.n_encoder_hidden)
            # MLP hidden sizes are 0 if not use_hidden_layers_in_specialization_mlp and we won't use an activation function
            arc_mlp_hidden = use_hidden_layers_in_specialization_mlp * (self.args.n_encoder_hidden + n_arc_mlp)//2
            rel_mlp_hidden = use_hidden_layers_in_specialization_mlp * (self.args.n_encoder_hidden + n_rel_mlp)//2
            activation = not use_hidden_layers_in_specialization_mlp
            if self.no_mlp_specialization:
                self.arc_mlp = MLP(n_in=self.args.n_encoder_hidden, n_hid=arc_mlp_hidden, n_out=n_arc_mlp, dropout=mlp_dropout, activation=activation)
                if self.use_rel_mlp:
                    self.rel_mlp = MLP(n_in=self.args.n_encoder_hidden, n_hid=rel_mlp_hidden, n_out=n_rel_mlp, dropout=mlp_dropout, activation=activation)
            else:
                self.arc_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_hid=arc_mlp_hidden, n_out=n_arc_mlp, dropout=mlp_dropout, activation=activation)
                self.arc_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_hid=arc_mlp_hidden, n_out=n_arc_mlp, dropout=mlp_dropout, activation=activation)
                if self.use_rel_mlp:
                    self.rel_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_hid=rel_mlp_hidden, n_out=n_rel_mlp, dropout=mlp_dropout, activation=activation)
                    self.rel_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_hid=rel_mlp_hidden, n_out=n_rel_mlp, dropout=mlp_dropout, activation=activation)
        
        original_arc_size = int(arc_size*arc_resize_factor)
        original_rel_size = int(rel_size*rel_resize_factor)
        arc_biaffine_in = self.args.n_encoder_hidden if self.use_word_embeddings_for_biaffine else n_arc_mlp
        rel_biaffine_in = n_rel_mlp if self.use_rel_mlp and not self.use_word_embeddings_for_biaffine else arc_biaffine_in

        if no_biaffine:
            self.arc_mlp_representation = MLP(n_in=arc_biaffine_in, n_hid=2*original_arc_size, n_out=original_arc_size)
        else:
            self.arc_attn = Biaffine(
                n_in=arc_biaffine_in, n_out=original_arc_size, scale=scale, bias_x=self.args.bias_x, bias_y=self.args.bias_y)
            if self.use_rel_biaffine:
                self.rel_attn = Biaffine(
                    n_in=rel_biaffine_in, n_out=original_rel_size, scale=scale, bias_x=self.args.bias_x, bias_y=self.args.bias_y)
                
            if self.use_biaffine_filter:
                orig_scorer_mlp_out = 50
                orig_scorer_mlp_hidden = (self.args.n_encoder_hidden + orig_scorer_mlp_out) // 2
                # MLP hidden sizes are 0 if not use_hidden_layers_in_specialization_mlp and we won't use an activation function 
                orig_scorer_mlp_hidden *= use_hidden_layers_in_specialization_mlp
                self.masked_arc_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_hid=orig_scorer_mlp_hidden, n_out=orig_scorer_mlp_out, dropout=mlp_dropout, activation=activation)
                self.masked_arc_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_hid=orig_scorer_mlp_hidden, n_out=orig_scorer_mlp_out, dropout=mlp_dropout, activation=activation)
                self.orig_arc_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_hid=orig_scorer_mlp_hidden, n_out=orig_scorer_mlp_out, dropout=mlp_dropout, activation=activation)
                self.orig_arc_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_hid=orig_scorer_mlp_hidden, n_out=orig_scorer_mlp_out, dropout=mlp_dropout, activation=activation)
                self.orig_rel_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_hid=orig_scorer_mlp_hidden, n_out=orig_scorer_mlp_out, dropout=mlp_dropout, activation=activation)
                self.orig_rel_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_hid=orig_scorer_mlp_hidden, n_out=orig_scorer_mlp_out, dropout=mlp_dropout, activation=activation)
                #Scoring Biaffines
                self.masked_arc_scorer = Biaffine(
                    n_in=orig_scorer_mlp_out, n_out=1, scale=scale, bias_x=self.args.bias_x, bias_y=self.args.bias_y)
                self.orig_arc_attn = Biaffine(
                    n_in=orig_scorer_mlp_out, n_out=1, scale=scale, bias_x=self.args.bias_x, bias_y=self.args.bias_y)
                self.orig_rel_attn = Biaffine(
                    n_in=orig_scorer_mlp_out, n_out=n_rels, bias_x=self.args.bias_x, bias_y=self.args.bias_y)

        if self.use_layer_filter:
            # Choose between a linear transformation or MLP for the filter
            n_in_scoring = original_arc_size
            n_hid_scoring = n_in_scoring//2
            if masked_arc_scorer == "lin":
                self.masked_arc_scorer = nn.Linear(n_in_scoring, 1)
            elif masked_arc_scorer == "mlp":
                self.masked_arc_scorer = MLP(n_in=n_in_scoring, n_hid=n_hid_scoring, n_out=1, dropout=filter_dropout_rate)

        self.arc_resizer = nn.Linear(original_arc_size, arc_size) if original_arc_size != arc_size else nn.Identity()
        if self.use_rel_biaffine:
            self.rel_resizer = nn.Linear(original_rel_size, rel_size) if original_rel_size != rel_size else nn.Identity()

            
        post_resizer2_arc_size = original_arc_size
        post_resizer2_rel_size = original_rel_size if self.use_rel_biaffine else post_resizer2_arc_size
        self.arc_resizer2 = nn.Linear(arc_size, post_resizer2_arc_size) if post_resizer2_arc_size != arc_size else nn.Identity()
        if self.use_rel_biaffine:
            self.rel_resizer2 = nn.Linear(rel_size, post_resizer2_rel_size) if post_resizer2_rel_size != rel_size else nn.Identity()

        # Choose between a linear transformation or MLP for scoring
        n_in_arc_scoring = round(post_resizer2_arc_size * (1-feature_scale if self.separate_features else 1))
        n_hid_arc_scoring = n_in_arc_scoring//2
        n_in_rel_scoring = round(post_resizer2_rel_size * (feature_scale if self.separate_features else 1))
        n_hid_rel_scoring = n_rels*2
        if self.separate_scoring:
            if arc_scorer == "lin":
                self.arc_scorer = nn.Linear(n_in_arc_scoring, 1)
            elif arc_scorer == "mlp":
                self.arc_scorer = MLP(n_in=n_in_arc_scoring, n_hid=n_hid_arc_scoring, n_out=1, dropout=scoring_dropout_rate)
            if rel_scorer == "lin":
                self.rel_scorer = nn.Linear(n_in_rel_scoring, n_rels)
            elif rel_scorer == "mlp":
                self.rel_scorer = MLP(n_in=n_in_rel_scoring, n_hid=n_hid_rel_scoring, n_out=n_rels, dropout=scoring_dropout_rate)
        else:
            if arc_scorer == "lin":
                self.scorer = nn.Linear(n_in_rel_scoring, n_rels+1)
            elif arc_scorer == "mlp":
                self.scorer = MLP(n_in=n_in_rel_scoring, n_hid=n_hid_rel_scoring, n_out=n_rels+1, dropout=scoring_dropout_rate)

        # Layernorms
        if self.use_layernorm:
            self.norm_arc_layer_before = LayerNorm(arc_size)
            self.norm_arc_layer_after = LayerNorm(arc_size)
            if self.use_rel_biaffine:
                self.norm_rel_layer_before = LayerNorm(rel_size)
                self.norm_rel_layer_after = LayerNorm(rel_size)
        
        # Choose between transformers
        if arc_transformer_choice == 0:
            self.arc_transformer = None
        elif arc_transformer_choice == 1:
            self.arc_transformer = TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=arc_size, dim_feedforward=4*arc_size,
                    nhead=n_heads_arcs, batch_first=True, norm_first=True,
                    dropout=transformer_encoder_dropout_rate),
                num_layers=n_layers_arcs
            )
        elif arc_transformer_choice == 2:
            self.arc_transformer = Kernel_transformer(
                use_cos=False,
                kernel="elu",
                layer_norm_choice=layer_norm_choice,
                flattened=flattened,
                dist_choice=dist_choice,
                use_causal_mask=use_causal_mask,
                d_model=arc_size,
                n_heads=n_heads_arcs,
                n_layers=n_layers_arcs,
                n_emb=200*200,
                ffn_ratio=4,
                rezero=False, ln_eps=1e-5, denom_eps=1e-5,
                bias=False, dropout=0.2, max_len=200*200, xavier=True
            )

        # If we use specific rel vectors, we'll use a specific rel transformer only if use_rel_transformer is True
        if self.use_rel_biaffine and not use_rel_transformer:
            self.rel_transformer = self.arc_transformer
        elif self.use_rel_biaffine and use_rel_transformer:
            if rel_transformer_choice == 0:
                self.rel_transformer = None
            elif rel_transformer_choice == 1:
                self.rel_transformer = TransformerEncoder(
                    GateTransformerEncoderLayer(
                        d_model=rel_size, dim_feedforward=4*rel_size,
                        nhead=n_heads_rels, batch_first=True, norm_first=True,
                        dropout=transformer_encoder_dropout_rate),
                    num_layers=n_layers_rels
                )
            elif rel_transformer_choice == 2:
                self.rel_transformer = Kernel_transformer(
                    use_cos=False,
                    kernel="elu",
                    layer_norm_choice=layer_norm_choice,
                    flattened=flattened,
                    dist_choice=dist_choice,
                    use_causal_mask=use_causal_mask,
                    d_model=rel_size,
                    n_heads=n_heads_rels,
                    n_layers=n_layers_rels,
                    n_emb=200*200,
                    ffn_ratio=4,
                    rezero=False, ln_eps=1e-5, denom_eps=1e-5,
                    bias=False, dropout=0.2, max_len=200*200, xavier=True
                )

    def get_selected_arcs(self, s_arc_filter, arc_seq_mask, lengths, lengths_arcs, arcs, batch_size, n, filter_factor):
        s_arc_filter = s_arc_filter.view((batch_size, n, n, 1)).squeeze(-1)
        k = min(n, filter_factor)
        cutoff = n*k
        lengths_arcs = torch.where(lengths < filter_factor, lengths ** 2, lengths * filter_factor)

        # create arc mask
        n_range = torch.arange(0,n).unsqueeze(0).expand(batch_size,n).cuda()
        len_range = torch.where(n_range < lengths.unsqueeze(-1).expand(batch_size,n), 1, 0)
        full_mask = (len_range.unsqueeze(-1) * len_range.unsqueeze(1))
        full_mask = torch.where(full_mask == 0, -1e8, 0)
        if self.training:
            # automatically choose gold arcs for each sentence
            golds = 1.1 * functional.one_hot(arcs, n)

        if self.use_gumbel_sampling:
            s_arc_filter = s_arc_filter.view(batch_size*n,n)
            # Warm-up
            s_arc_filter *= self.gumbel_beta if self.training else 1
            m = torch.distributions.gumbel.Gumbel(s_arc_filter, torch.tensor([self.gumbel_scale]).cuda())
            # Add Gumbel noise to filter scores
            s_arc_filter = m.sample(torch.tensor([1]).cuda()).squeeze(0).view(batch_size*n,n) if self.training else s_arc_filter
            # Apply softmax to noisy filter logits without out of bounds arcs
            s_arc_filter = torch.nn.Softmax(dim=1)(s_arc_filter+full_mask.view(batch_size*n,n)) #(BN,N)
            s_arc_filter_modified = s_arc_filter.detach().clone()
            if self.training:
                # Add gold arcs
                golds = golds.view(batch_size*n,n)
                s_arc_filter_modified = torch.where(golds > 1, golds, s_arc_filter_modified)
            selected_arcs = torch.topk(s_arc_filter_modified, k, dim=1).indices.cuda()
            hard = torch.nn.functional.one_hot(selected_arcs, num_classes=n)
            soft = s_arc_filter.unsqueeze(1).expand(batch_size*n,k,n)
            coef = hard + soft - soft.detach()
            selected_arcs = selected_arcs.view(batch_size, n, -1)
        # extract the correct mask filter
        arc_seq_mask = arc_seq_mask.view((batch_size, n, n)).gather(2, selected_arcs).cuda()
        if self.average_other_arcs and self.training and remaining_heads > 0:
            arc_seq_mask = torch.cat((arc_seq_mask, torch.ones_like(arc_seq_mask[...,-1].unsqueeze(-1))), -1)
        arc_seq_mask = arc_seq_mask.view((batch_size, cutoff))
       	return arc_seq_mask, selected_arcs, lengths_arcs, cutoff, coef

    
    def filter_arcs(self, arc_layers, rel_layers, arc_seq_mask, lengths, lengths_arcs, arcs, batch_size, n, filter_factor):
        arc_layers = arc_layers.view((batch_size, n, n, -1))
        arc_size = arc_layers.size(-1)
        if self.use_rel_biaffine:
            rel_layers = rel_layers.view((batch_size, n, n, -1))
            rel_size = rel_layers.size(-1)
        if self.learn_filter:
            s_arc_filter = self.masked_arc_scorer(arc_layers).squeeze(-1)
        else:
            s_arc_filter = self.arc_scorer(arc_layers).squeeze(-1)

        arc_seq_mask, selected_arcs, lengths_arcs, cutoff, coef = self.get_selected_arcs(
            s_arc_filter.clone(), arc_seq_mask, lengths, lengths_arcs, arcs, batch_size, n, filter_factor)

        # extract the vectors for the k best heads per word
        if self.use_gumbel_sampling:
            arc_layers = torch.einsum('bkn,bnd->bkd', coef, arc_layers.reshape(batch_size*n,n,arc_size)).view(batch_size, cutoff, arc_size)
            if self.use_rel_biaffine:
                rel_layers = torch.einsum('bkn,bnd->bkd', coef, rel_layers.reshape(batch_size*n,n,rel_size)).view(batch_size, cutoff, rel_size)

        return arc_layers, rel_layers, s_arc_filter, arc_seq_mask, selected_arcs, lengths_arcs, cutoff

    
    def combine_arcs(self, orig_arc_layers, arc_layers, orig_rel_layers, rel_layers, s_arc_filter, batch_size, n, filter_factor, cutoff, selected_arcs, mask):
        dim3 = cutoff//n
        # reshape layers
        selected_arcs = selected_arcs.reshape((batch_size,n, -1))
        arc_layers = arc_layers.view((batch_size, n, dim3, -1))
        arc_size = arc_layers.size(-1)
        if self.use_rel_biaffine:
            rel_layers = rel_layers.view((batch_size, n, dim3, -1))
            rel_size = rel_layers.size(-1)

        orig_arc_layers = orig_arc_layers.view((batch_size, n, n, -1))
        if self.average_other_arcs and self.training:
            orig_arc_layers = torch.ones_like(orig_arc_layers) * arc_layers[...,-1,:].unsqueeze(2)
        full_arcs = orig_arc_layers.scatter(2, selected_arcs.unsqueeze(-1).expand(-1, -1, -1, arc_size), arc_layers)
        if self.use_rel_biaffine:
            orig_rel_layers = orig_rel_layers.view((batch_size, n, n, -1))
            if self.average_other_arcs and self.training:
                orig_rel_layers = torch.ones_like(orig_rel_layers) * rel_layers[...,-1,:].unsqueeze(2)
            full_rels = orig_rel_layers.scatter(2, selected_arcs.unsqueeze(-1).expand(-1, -1, -1, rel_size), rel_layers)
        if self.separate_scoring:
            full_rels = full_rels if self.use_rel_biaffine else full_arcs
            s_arc = self.arc_scorer(full_arcs).squeeze(-1)
            s_rel = self.rel_scorer(full_rels)
        else:
            s_arc_rel = self.scorer(orig_arc_layers)
            s_arc = s_arc_rel[...,0].squeeze(-1)
            s_rel = s_arc_rel[...,1:]

        # mask filter scores
        s_arc_filter = s_arc_filter.view((batch_size, n, n))

        return s_arc, s_arc_filter, s_rel

    def forward(self, words, feats=None, arcs=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible arcs.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each arc.
        """

        n_arc_mlp = self.args.n_arc_mlp
        n_rel_mlp = self.args.n_rel_mlp
        use_biaffine_filter = self.use_biaffine_filter
        use_layer_filter = self.use_layer_filter
        learn_filter = self.learn_filter
        filter_factor = self.filter_factor
        arc_size = self.arc_size
        rel_size = self.rel_size
        arc_resize_factor = self.arc_resize_factor
        rel_resize_factor = self.rel_resize_factor
        arc_transformer_choice = self.arc_transformer_choice
        rel_transformer_choice = self.rel_transformer_choice
        separate_scoring = self.separate_scoring
        use_rel_transformer = self.use_rel_transformer
        feature_scale = self.feature_scale
        use_diffusion = self.use_diffusion
        no_biaffine = self.no_biaffine
        
        x = self.encode(words, feats)
        lengths = words.ne(self.args.pad_index).sum(1)
        # if bert is used, remove the added dimensions after the first
        if lengths.dim() == 2:
            lengths = lengths[:, :1].squeeze()
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)

        max_len = lengths.max()
        # lengths -> [batch_size, seq_len]
        lengths_arcs = lengths**2
        # lengths_arcs -> [batch_size, seq_len]

        if len(words.shape) == 2:
            batch_size, seq_len = words.shape
        else:
            batch_size, seq_len, fix_len = words.shape

        n = seq_len
        cutoff = n*n

        # [batch_size, seq_len]
        mask = words.ne(self.args.pad_index) if len(
            words.shape) < 3 else words.ne(self.args.pad_index).any(-1)

        # [batch_size, seq_len, seq_len]
        mask_dim1 = torch.unsqueeze(mask, 1)
        mask_dim2 = torch.unsqueeze(mask, 2)
        arc_seq_mask = mask_dim1 * mask_dim2
        arc_seq_mask = torch.where(arc_seq_mask == 0, False, True)
        arc_seq_mask = arc_seq_mask.view((batch_size, n * n))


        # Use word embeddings as the Biaffine input
        if self.use_word_embeddings_for_biaffine:
            arc_d, arc_h = x, x
            if self.use_rel_biaffine:
                rel_d, rel_h = x, x
        else:
            # Create MLP representations of words for each role, with or without normalization
            if self.use_norm_arc_mlp:
                x = self.mlp_norm(x)
            if self.no_mlp_specialization:
                arc_tmp = self.arc_mlp(x)
                arc_d, arc_h = arc_tmp, arc_tmp
                if self.use_rel_biaffine:
                    rel_tmp = self.rel_mlp(x) if self.use_rel_mlp else arc_tmp
                    rel_d, rel_h = rel_tmp, rel_tmp
            else:
                arc_d = self.arc_mlp_d(x)
                arc_h = self.arc_mlp_h(x)
                if self.use_rel_biaffine:
                    rel_d = self.rel_mlp_d(x) if self.use_rel_mlp else arc_d
                    rel_h = self.rel_mlp_h(x) if self.use_rel_mlp else arc_h

        s_arc_filter, selected_arcs = None, None
        if no_biaffine:
            # create arc_layers from arc_d and arc_h
            arc_d2 = arc_d.unsqueeze(1)
            arc_h2 = arc_h.unsqueeze(2)
            arc_layers = arc_d2+arc_h2
            arc_layers = self.arc_mlp_representation(arc_layers)
        else:
            if use_biaffine_filter:
                orig_s_arc = self.orig_arc_attn(self.orig_arc_mlp_d(x), self.orig_arc_mlp_h(x)).unsqueeze(1).permute(0, 2, 3, 1)
                orig_s_rel = self.orig_rel_attn(self.orig_rel_mlp_d(x), self.orig_rel_mlp_h(x)).permute(0, 2, 3, 1)
                s_arc_filter = self.masked_arc_scorer(self.masked_arc_mlp_d(x), self.masked_arc_mlp_h(x)).unsqueeze(1).permute(0, 2, 3, 1)
                arc_seq_mask, selected_arcs, lengths_arcs, cutoff, coef = self.get_selected_arcs(s_arc_filter, arc_seq_mask, lengths, lengths_arcs, arcs, batch_size, n, filter_factor)
                    
                # [batch_size, seq_len, filter_factor, arc_size*arc_resize_factor]
                if arc_size > 1:
                    arc_layers = self.arc_attn(arc_d, arc_h, selected_arcs).permute(0, 2, 3, 1)
                else:
                    arc_layers = self.arc_attn(arc_d, arc_h, selected_arcs).unsqueeze(1).permute(0, 2, 3, 1)
                if self.use_rel_biaffine:
                    if rel_size > 1:
                        rel_layers = self.rel_attn(rel_d, rel_h, selected_arcs).permute(0, 2, 3, 1)
                    else:
                        rel_layers = self.rel_attn(rel_d, rel_h, selected_arcs).unsqueeze(1).permute(0, 2, 3, 1)
            else:
                # [batch_size, seq_len, seq_len, arc_size*arc_resize_factor]
                if arc_size > 1:
                    arc_layers = self.arc_attn(arc_d, arc_h).permute(0, 2, 3, 1)
                else:
                    arc_layers = self.arc_attn(arc_d, arc_h).unsqueeze(1).permute(0, 2, 3, 1)
                if self.use_rel_biaffine:
                    if rel_size > 1:
                        rel_layers = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
                    else:
                        rel_layers = self.rel_attn(rel_d, rel_h).unsqueeze(1).permute(0, 2, 3, 1)
        
        n2 = filter_factor if use_biaffine_filter and n > filter_factor else n
        # [batch_size, seq_len * seq_len, arc_size*arc_resize_factor] if not use_biaffine_filter else [batch_size, seq_len * filter_factor, arc_size*arc_resize_factor]
        orig_arc_layers = arc_layers.reshape((batch_size, n * n2, -1))
        arc_layers = orig_arc_layers.clone()
        if self.use_rel_biaffine:
            orig_rel_layers = rel_layers.reshape((batch_size, n * n2, -1))
            rel_layers = orig_rel_layers.clone()
        else:
            orig_rel_layers = None
            rel_layers = None

        # Filter arcs
        if use_layer_filter:
            arc_layers, rel_layers, s_arc_filter, arc_seq_mask, selected_arcs, lengths_arcs, cutoff = self.filter_arcs(arc_layers, rel_layers, arc_seq_mask, lengths, lengths_arcs, arcs, batch_size, n, filter_factor)

        # [batch_size, seq_len * seq_len, arc_size] or [batch_size, cutoff, arc_size] if use_layer_filter
        arc_layers = self.arc_resizer(arc_layers)
        if self.use_rel_biaffine:
            rel_layers = self.rel_resizer(rel_layers)
        
        # Transformers
        if self.use_layernorm:
            arc_layers = self.norm_arc_layer_before(arc_layers)
            if self.use_rel_biaffine:
                rel_layers = self.norm_rel_layer_before(rel_layers)
        if arc_transformer_choice == 1:
            arc_layers = self.arc_transformer(arc_layers, src_key_padding_mask=arc_seq_mask.logical_not())
        elif arc_transformer_choice 2:
            arc_layers = self.arc_transformer(arc_layers, attention_mask=arc_seq_mask, lengths=lengths_arcs)
        if self.use_rel_biaffine:
            if rel_transformer_choice == 1
                rel_layers = self.rel_transformer(rel_layers, src_key_padding_mask=arc_seq_mask.logical_not())
            elif rel_transformer_choice == 2:
                rel_layers = self.rel_transformer(rel_layers, attention_mask=arc_seq_mask, lengths=lengths_arcs)
            
        if self.use_layernorm:
            arc_layers = self.norm_arc_layer_after(arc_layers)
            if self.use_rel_biaffine:
                rel_layers = self.norm_rel_layer_after(rel_layers)

        # [batch_size, seq_len * seq_len, arc_size*arc_resize_factor] or [batch_size, cutoff, arc_size*arc_resize_factor] if use_layer_filter
        arc_layers = self.arc_resizer2(arc_layers)
        if self.use_rel_biaffine:
            rel_layers = self.rel_resizer2(rel_layers)

        if use_layer_filter:
            s_arc, s_arc_filter, s_rel = self.combine_arcs(orig_arc_layers, arc_layers, orig_rel_layers, rel_layers, s_arc_filter, batch_size, n, filter_factor, cutoff, selected_arcs, mask)
            if not learn_filter:
                s_arc_filter = None
        else:
            n2 = filter_factor if use_biaffine_filter and n > filter_factor else n
            # [batch_size, seq_len, n2, arc_size*arc_resize_factor]
            arc_layers = arc_layers.reshape((batch_size, n, n2, -1))
            if self.use_rel_biaffine:
                rel_layers = rel_layers.reshape((batch_size, n, n2, -1))
            if self.separate_scoring:
                if self.separate_features:
                    feature_cutoff = round(arc_size*(1-feature_scale))
                    rel_layers = arc_layers[..., feature_cutoff:]
                    arc_layers = arc_layers[..., :feature_cutoff]
                else:
                    rel_layers = rel_layers if self.use_rel_biaffine else arc_layers
                # [batch_size, seq_len, n2]
                s_arc = self.arc_scorer(arc_layers).squeeze(-1)
                # [batch_size, seq_len, n2, n_rels]
                s_rel = self.rel_scorer(rel_layers)
            else:
                s_full = self.scorer(arc_layers)
                # [batch_size, seq_len, n2]
                s_arc = s_full[...,0]
                # [batch_size, seq_len, n2, n_rels]
                s_rel = s_full[...,1:]
            if use_biaffine_filter:
                s_arc = orig_s_arc.scatter(2, selected_arcs.unsqueeze(-1), s_arc.unsqueeze(-1)).squeeze(-1)
                s_rel = orig_s_rel.scatter(2, selected_arcs.unsqueeze(-1).expand(s_rel.shape), s_rel)
                s_arc_filter = s_arc_filter.squeeze(-1)
            else:
                selected_arcs = None

        return s_arc, s_rel, s_arc_filter, selected_arcs

    def loss(self, s_arc, s_rel, arcs, rels, mask, s_arc_filter=None, selected_arcs=None, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        arc_loss, rel_loss, filter_loss = 0, 0, 0
        if partial:
            mask = mask & arcs.ge(0)
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion_arc(s_arc, arcs) * 2*self.arc_rel_scale
        rel_loss = self.criterion_rel(s_rel, rels) * 2*(1-self.arc_rel_scale)
        if s_arc_filter is not None:
            s_arc_filter = s_arc_filter[mask]
            filter_arcs = torch.zeros_like(s_arc_filter).cuda()
            filter_arcs.scatter_add_(1, arcs.unsqueeze(-1), torch.ones_like(filter_arcs))
            filter_loss = self.filter_criterion(s_arc_filter, filter_arcs) * (1e-3 if self.use_layer_filter else 1)
        oracle = None
        if selected_arcs is not None:
            selected_arcs = selected_arcs[mask]
            retained_gold = torch.any(torch.eq(arcs[:, None], selected_arcs), dim=1).sum().item()
            total_gold = arcs.size(0)
            oracle = retained_gold, total_gold
        total_loss = arc_loss + rel_loss + filter_loss
        return total_loss, filter_loss, oracle

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False, selected_arcs=None):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        if self.reject_filtered_arcs:
            s_arc = s_arc - s_arc.min(-1).values.unsqueeze(-1)
            s_max = torch.where(s_arc.max(-1).values == 0, 1, s_arc.max(-1).values)
            s_arc = s_arc / s_max.unsqueeze(-1)
            zeros, ones = torch.zeros_like(s_arc), torch.ones_like(s_arc)
            mul = zeros.scatter(2, selected_arcs, ones)
            s_arc *= mul
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            arc_preds[bad] = (DependencyCRF if proj else MatrixTree)(s_arc[bad], mask[bad].sum(-1)).argmax
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds
