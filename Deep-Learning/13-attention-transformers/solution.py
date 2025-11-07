from transformers import AutoModelForSequenceClassification
from torch import nn
import torch
import math


class Embeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.dim,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.dim)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=config.eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings)
        """

        # Embedding the input ids
        input_embeds = self.word_embeddings(
            input_ids)  # (bs, max_seq_length, dim)
        seq_length = input_embeds.size(1)

        # Creating and embedding the position ids
        position_ids = torch.arange(
            seq_length, dtype=torch.long,
            device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(
            input_ids)  # (bs, max_seq_length)
        position_embeddings = self.position_embeddings(
            position_ids)  # (bs, max_seq_length, dim)

        # Compute the output embeddings
        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class AttentionBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # self-attention components

        # START YOUR CODE HERE #
        self.q_lin = nn.Linear(config.dim, config.all_head_size)
        self.k_lin = nn.Linear(config.dim, config.all_head_size)
        self.v_lin = nn.Linear(config.dim, config.all_head_size)
        # END YOUR CODE HERE #

        self.out_lin = nn.Linear(in_features=config.dim,
                                 out_features=config.dim,
                                 bias=True)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim,
                                          eps=config.eps)

        # feed-forward network
        self.lin1 = nn.Linear(in_features=config.dim,
                              out_features=3072,
                              bias=True)
        self.lin2 = nn.Linear(in_features=3072,
                              out_features=config.dim,
                              bias=True)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim,
                                              eps=config.eps)

    def forward(self, hidden_states):

        def shape(x):
            """ separate heads """
            return x.view(1, -1, 12, 64).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(1, -1, 12 * 64)

        bs = hidden_states.shape[0]
        n_nodes = hidden_states.shape[1]

        query = key = value = hidden_states
        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)

        # Separating the heads
        q = shape(q)  # (bs, n_heads, q_length, dim_per_head)
        k = shape(k)  # (bs, n_heads, k_length, dim_per_head)
        v = shape(v)  # (bs, n_heads, k_length, dim_per_head)

        # Normalizing the query-tensor
        q = q / math.sqrt(q.shape[-1])

        # START YOUR CODE HERE #

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(2, 3))

        # Transform the scores into probability distribution via softmax
        weights = nn.Softmax(dim=-1)(
            scores)  # (bs, n_heads, q_length, k_length)

        # Compute the weighted representation of the value-tensor (aka context)
        context = torch.matmul(weights,
                               v)  # (bs, n_heads, q_length, dim_per_head)

        # END YOUR CODE HERE #

        # Merging the heads again
        context = unshape(context)  # (bs, q_length, dim)

        # Additional projection of the context to get the output of the self-attention block
        sa_output = self.out_lin(context)
        sa_output = self.sa_layer_norm(sa_output + hidden_states)

        # Feed-forward network to compute the attention block output
        x = self.lin1(sa_output)
        x = nn.functional.gelu(x)
        ffn_output = self.lin2(x)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)

        return ffn_output, weights


class DistillBertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layers = config.n_layers

        # embedding
        self.embeddings = Embeddings(config)

        # encoder
        # START YOUR CODE HERE #
        self.attention_layers = torch.nn.Sequential(
            *[AttentionBlock(config) for i in range(config.n_layers)])
        # END YOUR CODE HERE #

        # classification
        self.pre_classifier = nn.Linear(in_features=config.dim,
                                        out_features=config.dim,
                                        bias=True)
        self.classifier = nn.Linear(in_features=config.dim,
                                    out_features=config.n_classes,
                                    bias=True)

        self.attention_probs = {i: [] for i in range(config.n_layers)}

    def forward(self, input_ids):
        """
        Parameters:
            input_ids (torch.Tensor): torch.tensor(bs, max_seq_length) The token ids to embed.
        Returns: torch.tensor(bs, n_classes) The computed logit scores for each class.
        """

        # Computing the embeddings
        hidden_states = self.embeddings(input_ids=input_ids).to(
            self.config.device)

        # Iteratively going through the attention layers
        encoder_input = hidden_states
        # START YOUR CODE HERE #
        for i, block in enumerate(self.attention_layers):

            output, attention_probs = block(encoder_input)

            self.attention_probs[i] = attention_probs
            encoder_input = output

        # END YOUR CODE HERE #

        # Pooling by selection the [CLS] token
        pooled_output = output[:, 0]  # (bs, dim)

        # Classification
        # START YOUR CODE HERE #
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)
        # END YOUR CODE HERE #

        return logits


def get_original_huggingface_model():
    distilbert = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english")
    _ = distilbert.eval()
    return distilbert


def test_embedding(your_embedding_module, input_ids):
    distilbert = get_original_huggingface_model()
    E1 = distilbert.base_model.embeddings(input_ids)

    your_embedding_module.load_state_dict(
        distilbert.base_model.embeddings.state_dict())
    E2 = your_embedding_module(input_ids)

    print(E1 == E2)


def test_block(config, your_block_module, embeds):
    torch.manual_seed(0)
    block = AttentionBlock(config)

    O1 = block(embeds)[0]
    O2 = your_block_module(embeds)[0]

    print(O1 == O2)


def test_model(your_model, input_ids):

    distilbert = get_original_huggingface_model()
    logits1 = distilbert(input_ids).logits

    logits2 = your_model(input_ids)

    print(logits1 == logits2)


def plot_attention(config, tokenizer, model, input_ids):
    import numpy as np
    import matplotlib.pyplot as plt

    subtokens = tokenizer.convert_ids_to_tokens(list(input_ids.squeeze()))
    A = np.array([
        model.attention_probs[i].detach().cpu().numpy().squeeze()
        for i in range(config.n_layers)
    ])
    A_avg = A.mean(1)

    for i in range(config.n_layers):
        f, ax = plt.subplots(1, 1)
        h = ax.imshow(A_avg[i], cmap='Reds', vmin=0, vmax=1)
        ax.set_xticks(range(len(subtokens)))
        ax.set_xticklabels(subtokens, rotation=45)

        ax.set_yticks(range(len(subtokens)))
        ax.set_yticklabels(subtokens)

        ax.set_title('layer {}'.format(i))
        plt.colorbar(h)
        plt.show()
