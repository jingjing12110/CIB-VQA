# @File :graph_transformer.py
# @Github :https://github.com/idiap/g2g-transformer
import math
import torch
import torch.nn as nn

from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import prune_linear_layer
from transformers.models.bert.modeling_bert import BertSelfOutput, \
    BertIntermediate, BertOutput, BertPooler, BertPreTrainedModel, \
    BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING

BertLayerNorm = torch.nn.LayerNorm


class BertGraphEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertGraphEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        
        if config.input_unlabel_graph:
            self.label_embeddings = nn.Embedding(
                config.label_size,
                config.hidden_size,
                padding_idx=0
            )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )
        
        self.LayerNorm = BertLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids=None, pos_ids=None,
                graph_rel=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None):
        """
        :param input_ids: token_ids, [bs, seq_len]
        :param pos_ids: Part-of-Speech ids of corresponding tokens
        :param graph_rel: [bs, seq_len]
        :param token_type_ids: sentence mask, [bs, seq_len]
        :param position_ids: position index of tokens, [bs, seq_len]
        :param inputs_embeds: token embedding, [bs, seq_len, embed_dim]
        :return: [bs, seq_len, embed_dim]
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None \
            else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            if pos_ids is not None:
                pos_embeds = self.word_embeddings(pos_ids)
                inputs_embeds = inputs_embeds + pos_embeds
        
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        if graph_rel is not None:
            label_embeddings = self.label_embeddings(graph_rel)
            embeddings = inputs_embeds + position_embeddings + label_embeddings \
                + token_type_embeddings
        else:
            embeddings = inputs_embeds + position_embeddings + \
                         token_type_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertGraphSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertGraphSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of "
                "the number of attention heads (%d)" % (
                    config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # graph input initialization
        self.input_label_graph = config.input_label_graph
        self.input_unlabel_graph = config.input_unlabel_graph
        self.layernorm_key = config.layernorm_key
        self.layernorm_value = config.layernorm_value
        
        if self.input_unlabel_graph or self.input_label_graph:
            if self.layernorm_key:
                self.layernorm_key_layer = nn.LayerNorm(self.attention_head_size)
            if self.layernorm_value:
                self.layernorm_value_layer = nn.LayerNorm(
                    self.attention_head_size)
        if self.input_label_graph:
            self.dp_relation_k = nn.Embedding(2 * config.label_size + 1,
                                              self.attention_head_size)
            self.dp_relation_v = nn.Embedding(2 * config.label_size + 1,
                                              self.attention_head_size)
        elif self.input_unlabel_graph:
            # pre-defined unlabeled graph:
            # 0: None
            # 1: spatial overlapping
            # 2: attribute-class embeddings closed
            # 3: both
            # self.dp_relation_k = nn.Embedding(4, self.attention_head_size)
            # self.dp_relation_v = nn.Embedding(4, self.attention_head_size)
            
            # KNN-Graph
            # 0: None
            # 1: has
            self.dp_relation_k = nn.Embedding(2, self.attention_head_size)
            self.dp_relation_v = nn.Embedding(2, self.attention_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def relative_matmul_dp(self, x, z):
        """ Helper function for dependency parsing relations"""
        x_t = x.transpose(1, 2)
        z_t = z.transpose(2, 3)
        out = torch.matmul(x_t, z_t)
        out = out.transpose(1, 2)
        return out
    
    def relative_matmul_dpv(self, x, z):
        """ Helper function for dependency parsing relations"""
        x = x.transpose(1, 2)
        out = torch.matmul(x, z)
        out = out.transpose(1, 2)
        return out
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            graph_arc=None
    ):
        mixed_query_layer = self.query(hidden_states)
        
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        if graph_arc is not None:
            dp_keys = self.dp_relation_k(graph_arc.to(key_layer.device))
            if self.layernorm_key:
                dp_keys = self.layernorm_key_layer(dp_keys)
            
            dp_values = self.dp_relation_v(graph_arc.to(key_layer.device))
            if self.layernorm_value:
                dp_values = self.layernorm_value_layer(dp_values)
        
        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if graph_arc is not None:
            attention_scores = attention_scores + self.relative_matmul_dp(
                query_layer, dp_keys)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers
            # in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        
        if graph_arc is not None:
            context_layer = context_layer + self.relative_matmul_dpv(
                attention_probs, dp_values)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        outputs = (context_layer, attention_probs
                   ) if self.output_attentions else (context_layer,)
        return outputs


class BertGraphAttention(nn.Module):
    def __init__(self, config):
        super(BertGraphAttention, self).__init__()
        self.self = BertGraphSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
    
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads,
                          self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads
        # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and
            # move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = \
            self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            graph_arc=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states,
            encoder_attention_mask, graph_arc
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertGraphLayer(nn.Module):
    def __init__(self, config):
        super(BertGraphLayer, self).__init__()
        self.attention = BertGraphAttention(config)
        
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertGraphAttention(config)
        
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            graph_arc=None
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, graph_arc)
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]
        
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask,
                encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:]
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


# ****************************************************************************
# adding for cross-modality fusion (myself)
class BertGraphCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertGraphCrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of "
                "the number of attention heads (%d)" % (
                    config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # graph input initialization
        self.input_label_graph = config.input_label_graph
        self.input_unlabel_graph = config.input_unlabel_graph
        self.layernorm_key = config.layernorm_key
        self.layernorm_value = config.layernorm_value

        if self.input_unlabel_graph or self.input_label_graph:
            if self.layernorm_key:
                self.layernorm_key_layer = nn.LayerNorm(self.attention_head_size)
            if self.layernorm_value:
                self.layernorm_value_layer = nn.LayerNorm(
                    self.attention_head_size)
        if self.input_label_graph:
            self.dp_relation_k = nn.Embedding(2 * config.label_size + 1,
                                              self.attention_head_size)
            self.dp_relation_v = nn.Embedding(2 * config.label_size + 1,
                                              self.attention_head_size)
        elif self.input_unlabel_graph:
            self.dp_relation_k = nn.Embedding(3, self.attention_head_size)
            self.dp_relation_v = nn.Embedding(3, self.attention_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None,
                output_attentions=False, graph_arc=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if graph_arc is not None:
            dp_keys = self.dp_relation_k(graph_arc.to(key_layer.device))
            if self.layernorm_key:
                dp_keys = self.layernorm_key_layer(dp_keys)
    
            dp_values = self.dp_relation_v(graph_arc.to(key_layer.device))
            if self.layernorm_value:
                dp_values = self.layernorm_value_layer(dp_values)
        
        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if graph_arc is not None:
            attention_scores = attention_scores + self.relative_matmul_dp(
                query_layer, dp_keys)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers
            # in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        if graph_arc is not None:
            context_layer = context_layer + self.relative_matmul_dpv(
                attention_probs, dp_values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs
                   ) if output_attentions else (context_layer,)
        return outputs


class BertGraphCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertGraphCrossAttention(config)
        self.output = BertOutput(config)
    
    def forward(self, input_tensor, ctx_tensor,
                ctx_att_mask=None, output_attentions=False,
                graph_arc=None):
        output = self.att(
            input_tensor, ctx_tensor, ctx_att_mask,
            output_attentions=output_attentions, graph_arc=graph_arc)
        
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (
            attention_output,)
        return outputs


class BertGraphXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # The cross-attention Layer
        self.visual_attention = BertGraphCrossAttentionLayer(config)
        
        # Self-attention Layers
        self.lang_self_att = BertGraphSelfAttention(config)
        self.visn_self_att = BertGraphSelfAttention(config)
        
        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)
    
    def cross_att(
        self,
        lang_input,
        lang_attention_mask,
        visual_input,
        visual_attention_mask,
        output_x_attentions=False,
        graph_arc=None,
    ):
        # Cross Attention
        lang_att_output = self.visual_attention(
            lang_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
            graph_arc=graph_arc
        )
        visual_att_output = self.visual_attention(
            visual_input,
            lang_input,
            ctx_att_mask=lang_attention_mask,
            output_attentions=False,
            graph_arc=graph_arc
        )
        return lang_att_output, visual_att_output

    def self_att(
            self,
            lang_input,
            lang_attention_mask,
            visual_input,
            visual_attention_mask,
            graph_arc=None,
    ):
        # Self Attention
        lang_att_output = self.lang_self_att(
            lang_input, lang_attention_mask,
            output_attentions=False, graph_arc=graph_arc)
        visual_att_output = self.visn_self_att(
            visual_input, visual_attention_mask,
            output_attentions=False, graph_arc=graph_arc)
        return lang_att_output[0], visual_att_output[0]

    def output_fc(self, lang_input, visual_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visual_inter_output = self.visn_inter(visual_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visual_output = self.visn_output(visual_inter_output, visual_input)

        return lang_output, visual_output
    
    def forward(
            self,
            lang_feats,
            lang_attention_mask,
            visual_feats,
            visual_attention_mask,
            output_attentions=False,
            graph_arc=None,
    ):
        lang_att_output, visual_att_output = self.cross_att(
            lang_input=lang_feats,
            lang_attention_mask=lang_attention_mask,
            visual_input=visual_feats,
            visual_attention_mask=visual_attention_mask,
            output_x_attentions=output_attentions,
            graph_arc=graph_arc
        )
        attention_probs = lang_att_output[1:]
        lang_att_output, visual_att_output = self.self_att(
            lang_att_output[0],
            lang_attention_mask,
            visual_att_output[0],
            visual_attention_mask,
            graph_arc=graph_arc
        )
        
        lang_output, visual_output = self.output_fc(
            lang_att_output, visual_att_output)
        return (
            (
                lang_output,
                visual_output,
                attention_probs[0],
            )
            if output_attentions
            else (lang_output, visual_output)
        )


# simple use of bert graph attention (original)
class BertGraphEncoder(nn.Module):
    def __init__(self, config):
        super(BertGraphEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        
        self.output_hidden_states = True
        self.layer = nn.ModuleList(
            [BertGraphLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            graph_arc=None
    ):
        """
        :param hidden_states:
        :param attention_mask:
        :param head_mask:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :param graph_arc: [bs, seq_len, seq_len]
        :return:
        """
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i],
                encoder_hidden_states, encoder_attention_mask, graph_arc
            )
            hidden_states = layer_outputs[0]
            
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without "
    "any specific head on top.",
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)
class BertGraphModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration
     (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``
        (batch_size, sequence_length, hidden_size)``
        Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``
        (batch_size, hidden_size)``
        Last layer hidden-state of the first token of the sequence
        (classification token) further processed by a Linear layer and
        a Tanh activation function. The Linear layer weights are trained from
        the next sentence prediction (classification) objective during Bert
        pretraining. This output is usually *not* a good summary
        of the semantic content of the input, you're often better
        with averaging or pooling
        the sequence of hidden-states for the whole input sequence.
        **hidden_states**:
            (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer +
            the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the
            initial embedding outputs.
        **attentions**: (`optional`, returned when
        ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``
            (batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute
            the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute",
        add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first
        element of the output tuple

    """
    
    def __init__(self, config):
        super(BertGraphModel, self).__init__(config)
        self.config = config
        self.embeddings = BertGraphEmbeddings(config)
        self.encoder = BertGraphEncoder(config)
        self.pooler = BertPooler(config)
        
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num:
                list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            pos_ids=None,
            graph_arc=None,
            graph_rel=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds "
                "at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None \
            else inputs_embeds.device
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long,
                                         device=device)
        
        # We can provide a self-attention mask of dimensions [batch_size,
        # from_seq_length, to_seq_length] ourselves in which case we just
        # need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition
            # to the padding mask
            # - if the model is an encoder, make the mask broadcastable
            # to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(
                    batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # not converting to long will cause errors with pytorch < 1.3
                causal_mask = causal_mask.to(torch.long)
                extended_attention_mask = causal_mask[:, None, :, :
                                          ] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask "
                "(shape {})".format(input_shape, attention_mask.shape))
        
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to
        # [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = \
                encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            
            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[
                                                  :, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[
                                                  :, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_"
                    "attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape))
            
            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = \
                (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape [bs x n_heads x N x N]
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                    -1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1,
                                             -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        embedding_output = self.embeddings(  # [bs, seq_len, 768]
            input_ids=input_ids,
            pos_ids=pos_ids,
            graph_rel=graph_rel,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds)
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            graph_arc=graph_arc
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # add hidden_states and attentions if they are here
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

