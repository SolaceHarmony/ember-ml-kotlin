# phonological_loop/models/phonological_loop_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Imports for Filter-less Architecture
from phonological_loop.features.analytic_features import AnalyticSignalExtractor
from phonological_loop.models.memory import PhonologicalLoopMemory
from phonological_loop.models.s4_layer import S4
from phonological_loop.models.classifiers import (
    SimpleMLPClassifier, DeepMLPClassifier, EnhancedClassifier
)

class PhonologicalLoopClassifier(nn.Module):
    # ──────────────────────────────────────────────────────────────────────
    # constructor signature unchanged (except removing noise filter params if they were added)
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        hop_length: int = 128,
        window_length: int = 512,
        buffer_len: int = 10,
        decay_factor: float = 0.9,
        num_recent_windows: int = 3,
        s4_d_model: int = 128,
        s4_d_state: int = 64,
        s4_l_max: int | None = None,
        s4_mode: str = "diag",
        s4_measure: str = "diag-lin",
        s4_dropout: float = 0.1,
        num_classes: int = 3,
        classifier_type: str = "simple",
        classifier_hidden_dim: int = 128,
        classifier_hidden_dims: list | None = None,
        classifier_dropout: float = 0.2,
        **_
    ):
        super().__init__()

        # ── 1. Analytic Feature Extractor ──────────────────────────────
        self.analytic_feature_extractor = AnalyticSignalExtractor(
            window_length=window_length,
            hop_length=hop_length
        )
        self._num_analytic_features = 5 # Based on AnalyticSignalExtractor output

        # ── 2. working-memory buffer ───────────────────────────────────
        self.memory = PhonologicalLoopMemory(
            feature_dim        = self._num_analytic_features,
            window_len         = None,        # infer on first batch
            buffer_len         = buffer_len,
            decay_factor       = decay_factor,
            num_recent_windows = num_recent_windows,
        )

        # pre-compute flattened memory output size ×once×
        # (will be finalised after we know window_len)
        self._mem_flat_dim = None

        # ── 3. optional S4 temporal mixer ──────────────────────────────
        self.use_s4  = True
        self.s4      = None
        self.d_model = s4_d_model                     # width fed to classifier

        # we’ll create S4 lazily once _mem_flat_dim is known
        self._s4_cfg = dict(
            d_state   = s4_d_state,
            l_max     = s4_l_max,
            mode      = s4_mode,
            measure   = s4_measure,
            dropout   = s4_dropout,
            transposed= True,
        )

        # ── 4. classifier head (built once dimensions settled) ─────────
        self.classifier_type      = classifier_type.lower()
        self.classifier_hidden_dim= classifier_hidden_dim
        self.classifier_hidden_dims= classifier_hidden_dims or [256,128,64]
        self.classifier_dropout   = classifier_dropout
        self.num_classes          = num_classes
        self.classifier           = None   # will be created later

    # ──────────────────────────────────────────────────────────────────────
    # internal utilities
    # ──────────────────────────────────────────────────────────────────────
    def _init_lazy_parts(self, T: int, device):
        """Finish building parts that need window_len (T)."""
        if self._mem_flat_dim is not None:
            return

        # Use the analytic feature dimension (5)
        feat_dim = self._num_analytic_features
        self.memory.window_len = T

        # Ensure memory buffers are allocated on the correct device
        if hasattr(self.memory, "_initialize_buffers"):
            self.memory._initialize_buffers(batch_size=1, device=device)
        elif hasattr(self.memory, "_allocate_buffers"):
            self.memory._allocate_buffers()

        self._mem_flat_dim = feat_dim * (self.memory.num_recent_windows + 1) * T

        # -- S4 ---------------------------------------------------------
        if self.use_s4:
            self.s4 = S4(
                d_model=self.d_model,
                **self._s4_cfg,
            ).to(device)

            proj_in  = self._mem_flat_dim
            proj_out = self.d_model
            self.mem2s4 = nn.Linear(proj_in, proj_out).to(device)

            # S4 output dimension is d_model
            self.post_s4 = None # No adapter needed

        # -- classifier -------------------------------------------------
        if self.use_s4:
            # Classifier receives d_model from S4 output
            cls_in = self.d_model
        else:
            cls_in = self._mem_flat_dim
        t = self.classifier_type
        if t == "simple":
            self.classifier = SimpleMLPClassifier(
                cls_in, self.num_classes,
                hidden_dim=self.classifier_hidden_dim
            ).to(device)

        elif t == "deep":
            self.classifier = DeepMLPClassifier(
                cls_in, self.num_classes,
                hidden_dims=self.classifier_hidden_dims,
                dropout=self.classifier_dropout
            ).to(device)

        else:  # default / "enhanced"
            self.classifier = EnhancedClassifier(
                cls_in, self.num_classes
            ).to(device)

    # ──────────────────────────────────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────────────────────────────────
    def forward(self, waveform: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        """
        Args
        ----
        waveform : (B, samples)

        Returns
        -------
        logits   : (B, num_classes)
        """
        # 1. Analytic Features (B, C=5, T=frames)
        analytic_feats = self.analytic_feature_extractor(waveform) # (B, 5, T)
        B, C, T = analytic_feats.shape # Use T (frames) now

        # one-time lazy init that needs T
        self._init_lazy_parts(T, analytic_feats.device)

        # 2. Working Memory
        # Pass RAW analytic features directly to memory
        mem = self.memory(analytic_feats) # (B, mem_flat)

        # 3. optional S4
        if self.use_s4:
            mem = self.mem2s4(mem)                     # (B, d_model)
            mem = mem.unsqueeze(-1)                    # (B, d_model, 1)
            mem, _ = self.s4(mem)
            mem = mem.squeeze(-1)                      # (B, _s4_out_dim)

            # No post_s4 adapter needed

        # 4. classify
        return self.classifier(mem)

    # ──────────────────────────────────────────────────────────────────────
    # public helpers (unchanged)
    # ──────────────────────────────────────────────────────────────────────
    def reset_state(self):
        self.memory.reset()

    # for backward compatibility
    def reset_memory(self):
        self.memory.reset()