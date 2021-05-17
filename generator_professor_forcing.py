import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import random
# import gensim
import datetime

class LSTMModel(nn.Module):
    def __init__(self, args, src_dict, dst_dict, use_cuda=True):
        super(LSTMModel, self).__init__()
        self.args = args
        self.use_cuda = use_cuda
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        self.fixed_max_len = args.fixed_max_len
        # Initialize encoder and decoder
        self.encoder = LSTMEncoder(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
        )
        self.decoder = LSTMDecoder(
            dst_dict,
            encoder_embed_dim=args.encoder_embed_dim,
            embed_dim=args.decoder_embed_dim,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            use_cuda=use_cuda
        )

    def forward(self, phase, epoch, sample):
        # encoder_output: (seq_len, batch, hidden_size * num_directions)
        # _encoder_hidden: (num_layers * num_directions, batch, hidden_size)
        # _encoder_cell: (num_layers * num_directions, batch, hidden_size)
        encoder_out = self.encoder(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])

        # # The encoder hidden is  (layers*directions) x batch x dim.
        # # If it's bidirectional, We need to convert it to layers x batch x (directions*dim).
        # if self.args.bidirectional:
        #     encoder_hiddens = torch.cat([encoder_hiddens[0:encoder_hiddens.size(0):2], encoder_hiddens[1:encoder_hiddens.size(0):2]], 2)
        #     encoder_cells = torch.cat([encoder_cells[0:encoder_cells.size(0):2], encoder_cells[1:encoder_cells.size(0):2]], 2)

        decoder_out, attn_scores, p, hidden_list = self.decoder(phase, epoch, self.fixed_max_len, sample['net_input']['prev_output_tokens'], encoder_out)
        decoder_out = F.log_softmax(decoder_out, dim=2)

        # sys_out_batch = decoder_out.contiguous().view(-1, decoder_out.size(-1))
        # loss = F.nll_loss(sys_out_batch, train_trg_batch, reduction='sum', ignore_index=self.dst_dict.pad())

        return decoder_out, p, hidden_list

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        vocab = net_output.size(-1)
        net_output1 = net_output.view(-1, vocab)
        if log_probs:
            return F.log_softmax(net_output1, dim=1).view_as(net_output)
        else:
            return F.softmax(net_output1, dim=1).view_as(net_output)


class LSTMEncoder(nn.Module):
    """LSTM encoder."""

    def __init__(self, dictionary, embed_dim=512, num_layers=1, dropout_in=0.1, dropout_out=0.1):
        super(LSTMEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            dropout=self.dropout_out,
            bidirectional=False,
        )

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # # pack embedded source tokens into a PackedSequence
        # packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        h0 = x.data.new(self.num_layers, bsz, embed_dim).zero_()
        c0 = x.data.new(self.num_layers, bsz, embed_dim).zero_()
        x, (final_hiddens, final_cells) = self.lstm(x, (h0, c0), )

        # unpack outputs and apply dropout
        # x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=0.)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, embed_dim]

        return x, final_hiddens, final_cells

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(2 * output_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        attn_scores = F.softmax(attn_scores.t(), dim=1).t()  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(nn.Module):
    def __init__(self, dictionary, encoder_embed_dim=512, embed_dim=512,
                 out_embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1, use_cuda=True):
        super(LSTMDecoder, self).__init__()
        self.use_cuda = use_cuda
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.layers = nn.ModuleList([
            LSTMCell(encoder_embed_dim + embed_dim if layer == 0 else embed_dim, embed_dim)
            for layer in range(num_layers)
        ])
        self.attention = AttentionLayer(encoder_embed_dim, embed_dim)
        if embed_dim != out_embed_dim:
            self.additional_fc = Linear(embed_dim, out_embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)


    def calculate_p(self, epoch, pre_label, pre_output):
        relativity_ = torch.cosine_similarity(pre_output, pre_label, dim=1)
        # oushi = torch.dist(pre_output, pre_label, p=2)
        relativity = relativity_.unsqueeze(-1)
        p_average = sum(relativity)/len(relativity)
        return p_average

    def forward(self, phase, epoch, fixed_max_len, prev_output_tokens, encoder_out, incremental_state=None, ):
        if phase == 'MLE':
            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
            bsz, seqlen = prev_output_tokens.size()
            # print("generator.py LSTMDecoder forward", seqlen)

            # get outputs from encoder
            encoder_outs, _, _ = encoder_out
            srclen = encoder_outs.size(0)

            x1 = self.embed_tokens(prev_output_tokens)  # (bze, seqlen, embed_dim)
            x2 = F.dropout(x1, p=self.dropout_in, training=self.training)
            embed_dim = x2.size(2)

            x3 = x2.transpose(0, 1)  # (seqlen, bsz, embed_dim)
            x = x3.detach()
            # initialize previous states (or get from cache during incremental generation)
            # cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
            # initialize previous states (or get from cache during incremental generation)
            cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')

            if cached_state is not None:
                prev_hiddens, prev_cells, input_feed = cached_state
            else:
                _, encoder_hiddens, encoder_cells = encoder_out
                num_layers = len(self.layers)
                prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
                prev_cells = [encoder_cells[i] for i in range(num_layers)]
                input_feed = x.data.new(bsz, embed_dim).zero_()

            attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
            outs = []
            hidden_list_MLE = []
            for j in range(fixed_max_len):
                input = torch.cat((x[j, :, :], input_feed), dim=1)
                for i, rnn in enumerate(self.layers):
                    # recurrent cell
                    hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                    # hidden state becomes the input to the next layer
                    input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                    # save state for next time step
                    prev_hiddens[i] = hidden
                    prev_cells[i] = cell
                hidden_list_MLE.append(torch.unsqueeze(hidden, 1))

                # apply attention using the last layer's hidden state
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
                out = F.dropout(out, p=self.dropout_out, training=self.training)
                # input feeding
                input_feed = out
                # save final output
                out_1 = out.unsqueeze(0)
                out_2 = out_1.transpose(1, 0)
                out_3 = self.fc_out(out_2)  # out_3 = [batch,1, num_vocab]
                outs.append(out_3)
                word = torch.argmax(out_3, dim=-1)  # word = [batch,1]
                out_4 = self.embed_tokens(word).squeeze(1)  # word = [batch,dim]
                if j < fixed_max_len - 1:
                    # p = self.calculate_p(epoch, x[j+1, :, :], out_4)
                    p == 0
                    is_teacher = random.random() > p
                    if not is_teacher:
                        x[j + 1, :, :] = out_4[:, :]

            hidden_MLE = torch.cat([hid for hid in hidden_list_MLE], 1)
            # cache previous states (no-op except during incremental generation)
            utils.set_incremental_state(
                self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

            attn_scores = attn_scores.transpose(0, 2)
            x = torch.cat(outs, dim=1).view(bsz, seqlen, -1)  # x = [batch,len,num_vocab]
            return x, attn_scores, p, hidden_MLE

        elif phase == 'PG':
            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
            bsz, seqlen = prev_output_tokens.size()
            # print("generator.py LSTMDecoder forward", seqlen)

            # get outputs from encoder
            encoder_outs, _, _ = encoder_out
            srclen = encoder_outs.size(0)

            x = self.embed_tokens(prev_output_tokens)  # (bze, seqlen, embed_dim)
            x = F.dropout(x, p=self.dropout_in, training=self.training)
            embed_dim = x.size(2)
            x = x.transpose(0, 1)  # (seqlen, bsz, embed_dim)
            # initialize previous states (or get from cache during incremental generation)
            # cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
            # initialize previous states (or get from cache during incremental generation)
            cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
            if cached_state is not None:
                prev_hiddens, prev_cells, input_feed = cached_state
            else:
                _, encoder_hiddens, encoder_cells = encoder_out
                num_layers = len(self.layers)
                prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
                prev_cells = [encoder_cells[i] for i in range(num_layers)]
                input_feed = x.data.new(bsz, embed_dim).zero_()

            attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
            outs = []
            hidden_list_PG = []
            for j in range(fixed_max_len):
                # input feeding: concatenate context vector from previous time step
                # teacher forcing
                # input_feed 是decoder hidden结合encoder output的attention向量
                # x 是input (prev_output_tokens)长度
                # print('11111111111111111111111', x.size(),input_feed.size())
                input = torch.cat((x[j, :, :], input_feed), dim=1)
                for i, rnn in enumerate(self.layers):
                    # recurrent cell
                    hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                    # hidden state becomes the input to the next layer
                    input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                    # save state for next time step
                    prev_hiddens[i] = hidden
                    prev_cells[i] = cell
                hidden_list_PG.append(torch.unsqueeze(hidden, 1))

                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
                out = F.dropout(out, p=self.dropout_out, training=self.training)
                # input feeding
                input_feed = out
                # save final output
                outs.append(out)
                if j < fixed_max_len - 1:
                    x[j + 1, :, :] = input[:, :]
            # cache previous states (no-op except during incremental generation)
            hidden_PG = torch.cat([hid for hid in hidden_list_PG], 1)  # hidden_list_PG=[batch,len,dim]
            utils.set_incremental_state(
                self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

            # collect outputs across time steps
            x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)
            # T x B x C -> B x T x C
            x = x.transpose(1, 0)
            # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
            attn_scores = attn_scores.transpose(0, 2)

            x = self.fc_out(x)
            p = 0
            return x, attn_scores, p, hidden_PG

        elif phase == 'test':
            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
            bsz, seqlen = prev_output_tokens.size()

            # get outputs from encoder
            encoder_outs, _, _ = encoder_out
            srclen = encoder_outs.size(0)

            x = self.embed_tokens(prev_output_tokens)  # (bze, seqlen, embed_dim)
            x = F.dropout(x, p=self.dropout_in, training=self.training)
            embed_dim = x.size(2)

            x = x.transpose(0, 1)  # (seqlen, bsz, embed_dim)

            # initialize previous states (or get from cache during incremental generation)
            # cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
            # initialize previous states (or get from cache during incremental generation)
            cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')

            if cached_state is not None:
                prev_hiddens, prev_cells, input_feed = cached_state
            else:
                _, encoder_hiddens, encoder_cells = encoder_out
                num_layers = len(self.layers)
                prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
                prev_cells = [encoder_cells[i] for i in range(num_layers)]
                input_feed = x.data.new(bsz, embed_dim).zero_()

            attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
            outs = []
            hidden_list_test = []
            for j in range(seqlen):
                # input feeding: concatenate context vector from previous time step
                # teacher forcing
                # input_feed 是decoder hidden结合encoder output的attention向量
                # x 是input (prev_output_tokens)长度
                input = torch.cat((x[j, :, :], input_feed), dim=1)

                for i, rnn in enumerate(self.layers):
                    # recurrent cell
                    hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                    # hidden state becomes the input to the next layer
                    input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                    # save state for next time step
                    prev_hiddens[i] = hidden
                    prev_cells[i] = cell

                # apply attention using the last layer's hidden state
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
                out = F.dropout(out, p=self.dropout_out, training=self.training)

                # input feeding
                input_feed = out

                # save final output
                outs.append(out)

            # cache previous states (no-op except during incremental generation)
            utils.set_incremental_state(
                self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

            # collect outputs across time steps
            x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)
            # T x B x C -> B x T x C
            x = x.transpose(1, 0)
            # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
            attn_scores = attn_scores.transpose(0, 2)

            x = self.fc_out(x)
            p = 0
            return x, attn_scores, p, hidden_list_test

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def reorder_incremental_state(self, incremental_state, new_order):
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        if not isinstance(new_order, Variable):
            new_order = new_order
        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m