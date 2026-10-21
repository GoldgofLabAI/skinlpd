# Standard library imports
import math

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F # for softmax   
import h5py

class GatedMILClassifier(nn.Module):
    def __init__(self, d_model=1024, num_classes=2, latent_dim=128, dropout=0.0,
                 attention_dropout=0.0, use_layer_norm=False, use_batch_norm=False,
                 use_skip_connection=False, tabular_dim=0):
        super(GatedMILClassifier, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.use_skip_connection = use_skip_connection
        # tabular_dim > 0 enables multimodal fusion: a tabular vector is concatenated
        # to the attention-pooled image embedding before the classifier. Default 0
        # keeps the classifier input at d_model, so existing checkpoints load unchanged.
        self.tabular_dim = tabular_dim
        clf_in = d_model + tabular_dim
        
        # Gated attention mechanism as small MLPs
        self.V = nn.Linear(d_model, latent_dim)
        self.U = nn.Linear(d_model, latent_dim)
        self.w = nn.Linear(latent_dim, 1)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.attention_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        
        # Normalization layers (helps with gradient flow and stability)
        if use_layer_norm:
            self.layer_norm_attn = nn.LayerNorm(latent_dim)
            self.layer_norm_pooled = nn.LayerNorm(d_model)
        
        if use_batch_norm:
            self.batch_norm_attn = nn.BatchNorm1d(latent_dim)
            self.batch_norm_pooled = nn.BatchNorm1d(d_model)
        
        # Skip connection projection (if dimensions don't match)
        if use_skip_connection and latent_dim != d_model:
            self.skip_proj = nn.Linear(d_model, latent_dim)
        
        # Final classifier with optional intermediate layer for regularization.
        # Input width = d_model (+ tabular_dim when multimodal fusion is enabled).
        if dropout > 0.3:  # Add intermediate layer for high dropout
            self.classifier = nn.Sequential(
                nn.Linear(clf_in, clf_in // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),  # Slightly less dropout in intermediate layer
                nn.Linear(clf_in // 2, num_classes)
            )
        else:
            self.classifier = nn.Linear(clf_in, num_classes)

        # Initialize weights for attention layers
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with proper scaling for better training stability."""
        # Xavier/Glorot initialization for attention layers
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.w.weight)
        
        # Initialize biases to zero
        if self.V.bias is not None:
            nn.init.zeros_(self.V.bias)
        if self.U.bias is not None:
            nn.init.zeros_(self.U.bias)
        if self.w.bias is not None:
            nn.init.zeros_(self.w.bias)
        
        # Initialize classifier weights
        if isinstance(self.classifier, nn.Linear):
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        else:  # Sequential classifier
            for module in self.classifier:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def compute_attention(self, F_features):
        """
        Compute gated attention with regularization.
        Input: [batch_size, num_F, d_model]
        Output: [batch_size, num_F]
        """
        # Apply dropout to input features
        F_features_dropped = self.dropout(F_features)
        
        # Gated attention computation
        K = torch.tanh(self.V(F_features_dropped))  # [batch_size, num_F, latent_dim]
        Q = torch.sigmoid(self.U(F_features_dropped))  # [batch_size, num_F, latent_dim]
        
        # Skip connection if enabled
        if self.use_skip_connection:
            if hasattr(self, 'skip_proj'):
                residual = self.skip_proj(F_features_dropped)
            else:
                residual = F_features_dropped
            K = K + residual
        
        # Apply normalization if enabled
        if self.use_layer_norm:
            K = self.layer_norm_attn(K)
        elif self.use_batch_norm:
            # Only apply batch norm if batch size > 1 (requirement for batch norm)
            batch_size, seq_len, features = K.shape
            if batch_size > 1:
                # BatchNorm expects [batch_size, features, seq_len], so we need to transpose
                K = K.transpose(1, 2)  # [batch_size, features, seq_len]
                K = self.batch_norm_attn(K)
                K = K.transpose(1, 2)  # [batch_size, seq_len, features]
            # For batch size = 1, skip batch norm (fallback gracefully)
        
        combined = K * Q  # [batch_size, num_F, latent_dim]
        scores = self.w(combined).squeeze(-1)  # [batch_size, num_F]
        
        # Apply attention dropout
        scores = self.attention_dropout(scores)
        
        attention_scores = F.softmax(scores, dim=1)  # [batch_size, num_F]
        return attention_scores

    def forward(self, F_features, tabular=None):
        attention_scores = self.compute_attention(F_features)  # [batch_size, num_F]
        pooled_features = torch.einsum('bif,bi->bf', F_features, attention_scores)  # [batch_size, d_model]

        # Apply normalization to pooled features if enabled (sized to d_model, so it
        # must run BEFORE tabular concatenation).
        if self.use_layer_norm:
            pooled_features = self.layer_norm_pooled(pooled_features)
        elif self.use_batch_norm:
            # Only apply batch norm if batch size > 1 (requirement for batch norm)
            if pooled_features.size(0) > 1:
                pooled_features = self.batch_norm_pooled(pooled_features)
            # For batch size = 1, skip batch norm (fallback gracefully)

        # Apply final dropout before classification
        pooled_features = self.dropout(pooled_features)

        # Multimodal fusion: concatenate the tabular covariate vector.
        if self.tabular_dim > 0:
            if tabular is None:
                raise ValueError("Model built with tabular_dim>0 but no tabular tensor passed to forward().")
            pooled_features = torch.cat([pooled_features, tabular], dim=1)  # [B, d_model+tabular_dim]

        class_scores = self.classifier(pooled_features)  # [batch_size, num_classes]
        return class_scores

    def compute_pooled_features(self, F_features):
        """
        Compute pooled features using attention scores.
        Input: [batch_size, num_F, d_model]
        Output: [batch_size, d_model]
        """
        attention_scores = self.compute_attention(F_features)  # [batch_size, num_F]
        pooled_features = torch.einsum('bif,bi->bf', F_features, attention_scores)  # [batch_size, d_model]
        return pooled_features.squeeze(0)
    
class MeanPoolClassifier(nn.Module):
    def __init__(self, d_model=1024, num_classes=2, latent_dim=512):
        super(MeanPoolClassifier, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Attention mechanism parameters
        self.W = nn.Parameter(torch.randn(latent_dim,))
        self.K = nn.Parameter(torch.randn(latent_dim, d_model))
        self.Q = nn.Parameter(torch.randn(latent_dim, d_model))
        
    def compute_attention(self, F_features):
        """
        Input is of shape [batch_size, num_F, 1024]
        Output is of shape [batch_size, num_F]
        """
        
        # Apply linear transformation K and Q
        K_transformed = torch.tanh(F_features @ self.K.transpose(0, 1))  # Shape: [batch_size, num_F, latent_dim]
        Q_transformed = torch.sigmoid(F_features @ self.Q.transpose(0, 1))  # Shape: [batch_size, num_F, latent_dim]
        
        # Element-wise multiplication
        combined_transforms = K_transformed * Q_transformed  # Shape: [batch_size, num_F, latent_dim]
        
        # Linear transformation W and take exponential
        attention_scores = combined_transforms @ self.W  # Shape: [batch_size, num_F]
        attention_scores = torch.exp(attention_scores)  # Shape: [batch_size, num_F]

        # Normalize attention scores
        attention_scores = attention_scores / attention_scores.sum(dim=1, keepdim=True)

        return attention_scores
    
    def forward(self, F_features):
        attention_scores = self.compute_attention(F_features)  # Shape: [batch_size, num_F]
        
        # Pool the features using attention scores
        pooled_features = torch.einsum('bif,bi->bf', F_features, attention_scores)  # [batch_size, 1024]
        
        # Pass the pooled_features through the classifier
        class_scores = self.classifier(pooled_features)  # Shape: [batch_size, num_classes]
        
        return class_scores

