# TGS Salt Identification Challenge

[Link to competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

Identify and classify toxic online comments.

## Model and Training

Used a 2-Layer BiLSTM (or BiGRU) with various pre-trained embeddings. Training 
a single model with ADAM took only a few epochs, but better results were obtained by 
ensembling among multiple embeddings and models (including linear models).

The primary data processing takes place in [toxic_pytorch_pooled_gru.ipynb](./toxic_pytorch_pooled_gru.ipynb).

The model looks like this:

```python
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nout, nlayers, dropemb=0.2, droprnn=0.0):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.drop = nn.Dropout2d(dropemb)
        
        self.ndir = 2 # if bidirectional else 1
        
        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid*self.ndir, nhid, 1, dropout=droprnn, bidirectional=True) for l in range(nlayers)]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid*self.ndir, nhid, 1, dropout=droprnn, bidirectional=True) for l in range(nlayers)]
        
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.decoder = nn.Linear(nhid*self.ndir*2, nout)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, lengths=None):
        # Pre-trained embedding, followed by dropout
        emb = self.encoder(input)
        
        raw_output = self.drop(emb)
        
        # Unpack padded sequence
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            raw_output = nn.utils.rnn.pack_padded_sequence(raw_output, lengths)
          
        # Core rnn feed forward
        for rnn in self.rnns:
            raw_output,_ = rnn(raw_output)
        
        # Repack padded sequences
        if lengths is not None:
            raw_output, lengths = nn.utils.rnn.pad_packed_sequence(raw_output)
        
        # Concatenate average and max pooling
        bsz = raw_output.size(1)
        rnn_avg = self.avg_pool(raw_output.permute(1,2,0))
        rnn_max = self.max_pool(raw_output.permute(1,2,0))
        rnn_out = torch.cat([rnn_avg.view(bsz,-1),rnn_max.view(bsz,-1)], dim=1)
        
        # Linear output layer
        return self.decoder(rnn_out)
```

Some embeddings were trained on the large corpus of wikipedia comments available.
Training for these embeddings is shown in the [toxic_gru_ext.ipynb](./toxic_gru_ext.ipynb) file.
