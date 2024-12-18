import math
import numpy as np
import tensorflow as tf


class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        STUDENT MUST WRITE:

        Computes attention given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys

        ## Fill triangle below diagonal of matrix with negative infinity and top part with 0.
        ## This helps to avoid over-contribution, since adjacency matrix is symmetric across diagonal. 
        ## Tile this upward to be compatible with addition against computed attention scores.
        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        atten_weights = tf.matmul(Q, K, transpose_b=True) / math.sqrt(K.get_shape()[2]) 
        # dimension is embedding_size
        
        if self.use_mask == True: 
            atten_weights += atten_mask
    
        atten_matrix = tf.nn.softmax(atten_weights)
        return atten_matrix
    
        # atten_weight shape [batch_size, window_size_query/num query, num keys]
        # softmax applied with axis = -1 -> to each row(query) ->  a single query across all key
        # to normalize the attention scores

        # TODO:
        # 1) compute attention weights using queries and key matrices 
        #       - if use_mask==True, then make sure to add the attention mask before softmax
        # 2) return the attention matrix

        # Check lecture slides for how to compute self-attention
        # Remember:
        # - Q is [batch_size x window_size_queries x embedding_size]
        # - K is [batch_size x window_size_keys x embedding_size]
        # - Mask is [batch_size x window_size_queries x window_size_keys]

        # Here, queries are matrix multiplied with the transpose of keys to produce for every query vector, weights per key vector.
        # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
        # Those weights are then used to create linear combinations of the corresponding values for each query.
        # Those queries will become the new embeddings. Return attention score as per lecture slides.


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # TODO:
        # Initialize the weight matrices for K, V, and Q.
        # They should be able to multiply an input_size vector to produce an output_size vector
        # Hint: use self.add_weight(...)
        self.K = self.add_weight(
            shape=(input_size, output_size),
            trainable=True, 
            name="K",
        )
        self.Q = self.add_weight(
            shape=(input_size, output_size),
            trainable=True, 
            name="Q",
        )
        self.V = self.add_weight(
            shape=(input_size, output_size),
            trainable=True, 
            name="V",
        )
        self.attn_mtx = AttentionMatrix(use_mask=self.use_mask)


        # input size represents the dimensionality of the embeddings 
        # output size -> desired (output_size vector)


    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # TODO:
        # - Apply 3 matrix products to turn inputs into keys, values, and queries. 
        # - You will need to use tf.tensordot for this.
        # - Call your AttentionMatrix layer with the keys and queries.
        # - Apply the attention matrix to the values.

        # -1 last axis of inputs, and 0 first exis of K 
        K = tf.tensordot(inputs_for_keys, self.K, axes=1)  
            # shape: [batch_size x KEY_WINDOW_SIZE x output_size]
        V = tf.tensordot(inputs_for_values, self.V, axes=1)
            # shape: [batch_size x KEY_WINDOW_SIZE x output_size]
        Q = tf.tensordot(inputs_for_queries, self.Q, axes=1)

        atten_mtx = self.attn_mtx((K, Q)) # ()

        result = atten_mtx @ V
        return result
            # shape: [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size]


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

        ## TODO: Add 3 heads as appropriate and any other necessary components

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        TODO: FOR CS2470 STUDENTS:

        This functions runs a multiheaded attention layer.

        Requirements:
            - Splits data for 3 different heads of size embed_sz/3
            - Create three different attention heads
            - Concatenate the outputs of these heads together
            - Apply a linear layer

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        return None


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # TODO:
        # 1) Define the Feed Forward, self-attention, encoder-decoder-attention, and layer normalization layers
        # 2) For 2470 students, use multiheaded attention

        self.ff_layer = tf.keras.Sequential([
            # tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(emb_sz, activation='relu'),
            ])

        self.self_atten         = AttentionHead(emb_sz, emb_sz, True)  if not multiheaded else MultiHeadedAttention(emb_sz, True)
        self.self_context_atten = AttentionHead(emb_sz, emb_sz, False) if not multiheaded else MultiHeadedAttention(emb_sz, False)
        # self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.layer_norm = tf.keras.layers.LayerNormalization()


    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        TODO:
        1) compute MASKED attention on the inputs
        2) residual connection and layer normalization
        3) computed UNMASKED attention using context
        4) residual connection and layer normalization
        5) feed forward layer
        6) residual layer and layer normalization
        7) return relu of tensor

        NOTES: This article may be of great use:
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """

        masked_att = self.self_atten(inputs, inputs, inputs) #  key values query
        masked_att += inputs # self.add = tf.keras.layers.Add() tk
        normalized_masked_att = self.layer_norm(masked_att)
        # print(normalized_masked_att)

        unmasked_att = self.self_context_atten(context_sequence, context_sequence, normalized_masked_att) 
        # tk which query
        unmasked_att += masked_att # adding the previous thing? tk
        unmasked_att = self.layer_norm(unmasked_att)
        
        ff_att = self.ff_layer(unmasked_att)
        # print(ff_att)
        ff_att += unmasked_att
        ff_att = self.layer_norm(ff_att)
        # print(ff_att)

        return tf.nn.relu(ff_att)


def positional_encoding(length, depth):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    ## TODO: Can remove signature
    depth = depth/2
    ## Generate a range of positions and depths 
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size

        ## TODO: Implement Component

        ## Embed labels into an optimizable embedding space
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        # tk embed size / hidden size -> in decoder

        ## Implement sinosoidal positional encoding: offset by varying sinosoidal frequencies. 
        ## HINT: May want to use the function above...
        self.pos_encoding = positional_encoding(window_size, embed_size) #tk length=2048, window 20

    def call(self, x):
        # length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))

        # x = x + self.pos_encoding[tf.newaxis, :length, :]
        x = x + self.pos_encoding

        ## TODO: Get embeddings and and scale them by sqrt of embedding size, and add positional encoding.
        return x
    