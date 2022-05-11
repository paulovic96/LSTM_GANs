import torch 

class LSTM_Critic(torch.nn.Module):
    def __init__(self, input_size=30,
                 embed_size = 300, 
                 output_size=1,
                 hidden_size=200,
                 num_lstm_layers=2,
                 dropout=0.5):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size + embed_size, hidden_size, num_layers=num_lstm_layers, batch_first = True, dropout=dropout)
        self.fully_connected = torch.nn.Linear(hidden_size,output_size)

    def forward(self, x, lens, vector, *args):
        
        x = torch.cat([x, vector.unsqueeze(1).repeat(1, x.shape[1], 1)],dim=2)
        output, (h_n, _) = self.lstm(x)
        output = torch.stack([output[i, (last - 1).long(), :] for i, last in enumerate(lens)])
        output = self.fully_connected(output)
        # average pooling
        #output = output.mean([1])
        
        return output


class LSTM_Generator(torch.nn.Module):
    def __init__(self,channel_noise = 60,
                 embed_size = 300,
                 output_size=30,
                 hidden_size=200,
                 num_lstm_layers=2,
                 dropout=0.5,
                 activation = torch.nn.LeakyReLU(0.2)):
        super().__init__()
        
        self.output_activation = torch.nn.Tanh()
        self.activation = activation
        self.fully_connected = torch.nn.Linear(channel_noise + embed_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers=num_lstm_layers, batch_first = True, dropout=dropout)
        self.post_linear = torch.nn.Linear(hidden_size, output_size)


    def forward(self, x, lens, vector, *args):
        x = torch.cat([x,vector.unsqueeze(1).repeat(1,x.shape[1],1)], dim = 2)
        output = self.fully_connected(x)
        output = self.activation(output)
        output, _ = self.lstm(output)
        #output = torch.stack([output[i, (last - 1).long(), :] for i, last in enumerate(lens)])
        
        output = self.post_linear(output)
        output = self.output_activation(output)

        return output
