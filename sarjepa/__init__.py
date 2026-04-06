# Vendored SAR-JEPA code from https://github.com/waterdisappear/SAR-JEPA
# License: CC BY-NC 4.0 (non-commercial use only; must train own weights for TerraLatent commercial use)
#
# ============================================================================
# SAR-JEPA INTERNALS ANALYSIS
# ============================================================================
#
# 1. MODEL ARCHITECTURE & CONSTRUCTORS
# -------------------------------------
# TWO separate model classes:
#
# A) PRETRAINING MODEL (models_lomar.py):
#    Class: MaskedAutoencoderViT
#    Constructor: MaskedAutoencoderViT(img_size=224, patch_size=16, in_chans=1,
#                   embed_dim=1024, depth=24, num_heads=16,
#                   decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#                   mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False)
#    Factory functions:
#      - mae_vit_base_patch16: embed_dim=768, depth=12, num_heads=12  <-- DEFAULT used in pretraining
#      - mae_vit_large_patch16: embed_dim=1024, depth=24, num_heads=16
#      - mae_vit_tiny: embed_dim=192, depth=12, num_heads=3
#      - mae_vit_small: embed_dim=384, depth=12, num_heads=6
#    Note: Invoked as models_lomar.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
#
# B) FINETUNING MODEL (modeling_finetune.py / models_vit_rp.py):
#    Class: VisionTransformer (from modeling_finetune.py, BEiT-style with init_values, rel_pos_bias)
#    models_vit_rp.py wraps it with global_pool support.
#    Key difference: uses relative position bias (window_size based) instead of iRPE.
#
# 2. INPUT FORMAT
# ---------------
# - in_chans=1 (SINGLE CHANNEL, not 3-channel RGB)
# - Default img_size=224, patch_size=16 -> 14x14 = 196 patches
# - Input shape: (B, 1, 224, 224)
# - Pretraining uses torchvision transforms: RandomResizedCrop(224), RandomHorizontalFlip(),
#   ColorJitter(contrast=0.5), ToTensor() (NO normalization)
# - For our 64x64 patches: need to either resize to 224 or change img_size/patch_size.
#   Options: (a) use img_size=64, patch_size=8 -> 8x8=64 patches,
#            (b) use img_size=128, patch_size=16 -> 8x8=64 patches (better match to pretrained weights)
#
# 3. WEIGHT LOADING
# -----------------
# Checkpoint format: checkpoint['model'] contains state_dict
# Loading in finetune:
#   checkpoint = torch.load(path, map_location='cpu')
#   checkpoint_model = checkpoint['model']
#   model.load_state_dict(checkpoint_model, strict=False)
# Head keys ('head.weight', 'head.bias') are removed if shape mismatch.
#
# For ENCODER-ONLY extraction from pretrained MaskedAutoencoderViT:
#   The encoder keys are: patch_embed.*, cls_token, pos_embed, blocks.*, norm.*
#   The decoder keys (to skip): encoder_pred.*, decoder_blocks.*, decoder_norm.*, decoder_pred.*, mask_token
#   The GF feature extractors: sarfeature1-4.* (buffers, no learned params)
#
# 4. FORWARD METHOD SIGNATURES
# ----------------------------
# Pretraining (MaskedAutoencoderViT):
#   forward(imgs, window_size=7, num_window=4, mask_ratio=0.8) -> (loss, pred, mask_indices)
#   forward_encoder(x, window_size, num_window, mask_ratio) -> (latent, mask_indices, ids_restore)
#
# For FEATURE EXTRACTION (our use case), we need a custom forward that:
#   1. patch_embed(x) -> (B, num_patches, embed_dim)
#   2. Add cls_token + pos_embed
#   3. Pass through self.blocks
#   4. self.norm(x)
#   5. Return x[:, 0] (cls token) OR x[:, 1:].mean(1) (mean pool) -> (B, embed_dim)
#   Output dim: 768 for vit_base
#
# Finetune (VisionTransformer):
#   forward(x) -> logits
#   forward_features(x) -> cls token embedding or mean-pooled embedding
#   Uses iRPE (image relative position encoding) from irpe.py with product method on keys
#
# 5. MASKING (LoMaR - Local Masked Reconstruction)
# -------------------------------------------------
# NOT standard MAE random masking. Uses LOCAL WINDOW masking:
#   - Samples num_window (default 4) random windows of window_size (default 7) patches
#   - Within each window, masks mask_ratio (default 0.8) of patches
#   - Only reconstructs within local windows (more efficient than global)
#   - This is the key innovation: Local Masked Auto-Reconstruction
#
# 6. GRADIENT FEATURE EXTRACTORS (GF class)
# ------------------------------------------
# Multi-scale gradient features used as reconstruction targets (NOT pixel reconstruction):
#   - 4 GF instances with kernel sizes: 5, 9, 13, 17
#   - Each GF computes log-ratio gradient magnitudes (SAR-specific, handles multiplicative noise)
#   - forward_loss concatenates all 4 patchified gradient maps as target: (B, L, 256*4)
#   - Decoder output dim is 256*4 = 1024 to match
#   - Formula: norm(log(sum_left/sum_right), log(sum_top/sum_bottom))
#   - These are BUFFERS (no learned params), purely geometric operations
#   - This is SAR-specific: handles multiplicative speckle noise better than pixel-space targets
#
# 7. iRPE (image Relative Position Encoding)
# -------------------------------------------
# From irpe.py, used in the Attention layers of vision_transformer_irpe.py:
#   - Config: ratio=1.9, method="product", mode='ctx' (contextual), shared_head=True, skip=1, rpe_on='k'
#   - Applied only on keys (not queries or values)
#   - Uses piecewise index function for bucketing relative positions
#   - Requires easydict package
#   - Optional CUDA-optimized rpe_ops (falls back gracefully if not built)
#
# 8. KEY DEPENDENCIES
# -------------------
# - torch, timm (0.3.2 for finetune, 0.9.12 works for pretrain)
# - easydict (for irpe.py config)
# - The rpe_ops CUDA extension is optional (Python fallback exists)
#
# ============================================================================
