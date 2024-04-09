import tensorflow as tf

try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers

        # Define feed forward layer(s) to embed image features into a vector 
        # with the models hidden size
        self.image_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu), 
            tf.keras.layers.Dense(self.hidden_size, activation='relu')])

        # Define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.hidden_size)
        # self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size)
        # Define decoder layer that handles language and image context:     
        self.decoder = tf.keras.layers.GRU(units=self.hidden_size, return_sequences=True, return_state=False)
        # return the full sequence of outputs

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation=tf.nn.leaky_relu), 
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(self.vocab_size)])
        # can add multiple layers


    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension for initial state
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder 
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        image_embed = self.image_embedding(encoded_images)
        caption_embed = self.embedding(captions)
        decoder_output = self.decoder(inputs=caption_embed, initial_state=image_embed)
        logits = self.classifier(decoder_output)
        return logits

########################################################################################

class TransformerDecoder(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer(s) to embed image features into a vector 
        self.image_embedding = tf.keras.Sequential([
            # tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu), 
            tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu), 
            tf.keras.layers.Dense(self.hidden_size, activation='relu'), 
            tf.keras.layers.Dense(self.hidden_size)
            ]) # relu - making negative -> zero
        
        # Define positional encoding to embed and offset layer for language:
        # vocab_size, embed_size, window_size
        self.encoding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)
        
        # Define transformer decoder layer:
        self.decoder = TransformerBlock(emb_sz=self.hidden_size)

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation=tf.nn.leaky_relu), 
            tf.keras.layers.Dense(128, activation='relu'), 
            tf.keras.layers.Dense(self.vocab_size)])

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate **logits**
        image_embed = self.image_embedding(tf.expand_dims(encoded_images, 1))
        caption_embed = self.encoding(captions)
        decoder_output = self.decoder(inputs=caption_embed, context_sequence=image_embed)
        logits = self.classifier(decoder_output)
        return logits


