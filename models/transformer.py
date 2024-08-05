from torch import nn
import torch



# feature processing
class MLPEmbedder(nn.Module):
    def __init__(self,
                 input_dim, # number of features
                 embed_dim, # number of features after embedding
                 mlp_embed_factor=2,
                 nonlin_func=nn.ReLU(), 
                 pdropout=0.3, 
                 num_encoder_units=2):
        
        super().__init__()
        
        self.We = nn.Linear(input_dim, embed_dim, bias=True)
        encunit_layers = [MLPBlock(embed_dim,
                                   embed_dim,
                                   mlp_embed_factor,
                                   nonlin_func, 
                                   pdropout)
                          for i in range(num_encoder_units)]

        self.encunit_pipeline = nn.Sequential(*encunit_layers)

    def forward(self, X):
        """
        Args:
            X: tensor, float32, (batch, embed_dim) representing x_target
        """

        X = self.We(X)
        out = self.encunit_pipeline(X)
        return out

class MLPBlock(nn.Module):
            
    def __init__(self,
                 input_dim,
                 embed_dim,
                 mlp_embed_factor,
                 nonlin_func, 
                 pdropout):
        
        super().__init__()
        
        assert input_dim == embed_dim

        self.layernorm_1 = nn.LayerNorm(embed_dim)

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, embed_dim*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_dim*mlp_embed_factor, embed_dim)
        )
        self.dropout = nn.Dropout(p=pdropout)

    def forward(self, X):
        """
        Args:
            X: input tensor, (batch, sequence length, input_dim)
        """
        o = self.MLP(X)
        o = self.layernorm_1(o + X)
        o = self.dropout(o)
        return o

# meta learner for the RNN and MLP ensemble
class MLPDecoder(nn.Module):
    def __init__(self,
                 inp_dim,
                 embed_dim,
                 outp_dim,
                 mlp_embed_factor=2,
                 nonlin_func=nn.ReLU(), 
                 pdropout=0.3, 
                 num_encoder_units=2):
        
        super().__init__()
        
        self.We = nn.Linear(inp_dim, embed_dim, bias=True)
        encunit_layers = [MLPBlock(embed_dim,
                                   embed_dim,
                                   mlp_embed_factor,
                                   nonlin_func, 
                                   pdropout)
                          for i in range(num_encoder_units)]

        self.encunit_pipeline = nn.Sequential(*encunit_layers)

        self.W_mu = nn.Linear(embed_dim, outp_dim)
        # self.W_sigma = nn.Linear(embed_dim, outp_dim)
        
        # self.softplus = nn.Softplus()

    def forward(self, X):
        """
        Args:
            X: tensor, float32, (batch, embed_dim) representing x_target
        """

        X = self.We(X)
        out = self.encunit_pipeline(X)

        mu = self.W_mu(out)
        return mu
        # logsigma  = self.W_sigma(out)
        # sigma = 0.1 + 0.9 * self.softplus(logsigma)

        # return mu, sigma