from imports import *

class FeatureGroupTransformerModel(nn.Module):
    def __init__(self, feature_groups, output_size, task_type='binary'):
        super().__init__()
        self.feature_groups = feature_groups
        self.group_names = list(feature_groups.keys())
        self.num_groups = len(self.group_names)
        self.task_type = task_type
        self.embedding_dim = 128
        self.num_heads = 16
        self.num_encoder_layers = 3
        self.feature_encoders = nn.ModuleDict()
        for group_name in self.group_names:
            _, group_size = feature_groups[group_name]
            self.feature_encoders[group_name] = nn.Sequential(
                nn.Linear(group_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.embedding_dim)
            )
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_groups, self.embedding_dim) * 0.1)
        self.group_type_embedding = nn.Parameter(torch.randn(1, self.num_groups, self.embedding_dim) * 0.1)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=128,
            dropout=0.3,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=self.num_encoder_layers,
            norm=nn.LayerNorm(self.embedding_dim)
        )
        self.attention_scorer = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.output_layers = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )
        if task_type == 'binary':
            self.output_activation = nn.Sigmoid()
        elif task_type == 'ternary':
            self.output_activation = None
        else:
            self.output_activation = nn.Sigmoid()


    def forward(self, x, return_attention_weights=False):
        batch_size = x.size(0)
        group_embeddings = []
        for group_name in self.group_names:
            start_idx, group_size = self.feature_groups[group_name]
            group_features = x[:, start_idx : start_idx + group_size]
            group_embedding = self.feature_encoders[group_name](group_features)
            group_embeddings.append(group_embedding)
        encoded_groups = torch.stack(group_embeddings, dim=1)
        encoded_groups = encoded_groups + self.pos_encoding + self.group_type_embedding
        transformer_output = self.transformer_encoder(encoded_groups)
        attn_scores = self.attention_scorer(transformer_output)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled_output = torch.sum(transformer_output * attn_weights, dim=1)
        final_output = self.output_layers(pooled_output)
        if self.output_activation is not None:
            final_output = self.output_activation(final_output)
        if return_attention_weights:
            return final_output, attn_weights.squeeze(-1)
        else:
            return final_output