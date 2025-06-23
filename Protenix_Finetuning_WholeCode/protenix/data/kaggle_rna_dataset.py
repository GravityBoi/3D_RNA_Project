# Copyright 2024 ByteDance and/or its affiliates.
# Licensed under the Apache 2.0 licence.

"""
Kaggle Stanford-RNA-3D folding dataset interface for Protenix.
Incorporates Protenix-style cropping.
"""

from __future__ import annotations

import os
import json
import traceback
import logging
from collections import defaultdict
from typing import Any, Mapping, Optional, Tuple, Dict
import sys 

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from biotite.structure import AtomArray # For type hinting

# Protenix imports
from protenix.utils.distributed import DIST_WRAPPER
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.utils import make_dummy_feature, data_type_transform
from protenix.data.msa_featurizer import MSAFeaturizer
from protenix.data.tokenizer import TokenArray
from protenix.utils.torch_utils import dict_to_tensor
from protenix.utils.cropping import CropData # For Protenix-style cropping
from protenix.data.featurizer import Featurizer as ProtenixFeaturizer # To generate final features on cropped data

# ----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------- util to build loaders ------------------------------------------
def _make_loader(dataset: "_SimpleRNADataset", shuffle: bool, nw: int) -> DataLoader:
    if DIST_WRAPPER.world_size > 1:
        sampler = DistributedSampler(dataset,
                                     num_replicas=DIST_WRAPPER.world_size,
                                     rank=DIST_WRAPPER.rank,
                                     shuffle=shuffle)
    else:
        sampler = None
    return DataLoader(dataset,
                      batch_size=1,
                      shuffle=(sampler is None and shuffle),
                      sampler=sampler,
                      num_workers=nw,
                      collate_fn=lambda b: b[0],
                      drop_last=False)
# ----------------------------------------------------------------------------

def get_kaggle_rna_dataloaders(cfg, msa_featurizer_train=None, msa_featurizer_val=None) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    kcfg = cfg.data.kaggle_rna3d # This kcfg will be passed to _SimpleRNADataset
    root = kcfg.root

    train_ds = _SimpleRNADataset(
        kcfg=kcfg, # Pass the specific kaggle_rna3d config part
        sequences_csv_path=os.path.join(root, kcfg.train_sequences_csv),
        labels_csv_path=os.path.join(root, kcfg.train_labels_csv),
        is_validation=False,
        msa_featurizer=msa_featurizer_train,
        apply_temporal_cutoff=APPLY_TEMPORAL_CUTOFF_FOR_TRAIN,
        temporal_cutoff_date=TEMPORAL_CUTOFF_DATE_STR
    )
    val_ds = _SimpleRNADataset(
        kcfg=kcfg, # Pass the specific kaggle_rna3d config part
        sequences_csv_path=os.path.join(root, kcfg.val_sequences_csv),
        labels_csv_path=os.path.join(root, kcfg.val_labels_csv),
        is_validation=True,
        msa_featurizer=msa_featurizer_val,
        apply_temporal_cutoff=False, # No cutoff for validation set itself
        temporal_cutoff_date=None
    )

    # nw = getattr(cfg.data, "num_dl_workers", getattr(cfg, "num_workers", 4))
    nw = 0
    return _make_loader(train_ds, True, nw), {"kaggle_val": _make_loader(val_ds, False, nw)}

# --------------------------------------------------------------------------- #
# 1.  “Heavy” dataset that does the csv parsing and featurisation
# --------------------------------------------------------------------------- #
DEBUG_FEATURIZE = False 
PRE_CROP_THRESHOLD = 500 
PRE_CROP_TARGET_LENGTH = 300 

APPLY_TEMPORAL_CUTOFF_FOR_TRAIN = True 
# The cutoff date (YYYY-MM-DD). Samples with temporal_cutoff >= this date will be excluded from training.
TEMPORAL_CUTOFF_DATE_STR = "2024-01-01"

class _SimpleRNADataset(Dataset):

    def __init__(
        self,
        kcfg: Mapping[str, Any], 
        sequences_csv_path: str,
        labels_csv_path: str,
        is_validation: bool = False,
        msa_featurizer: Optional[MSAFeaturizer] = None,
        apply_temporal_cutoff: bool = False,
        temporal_cutoff_date: Optional[str] = None
    ):
        super().__init__()
        self.is_validation = is_validation
        self.msa_featurizer = msa_featurizer
        self.apply_temporal_cutoff = apply_temporal_cutoff
        self.temporal_cutoff_date = temporal_cutoff_date

        # self.num_samples_to_load = kcfg.get("num_train_subset") if not is_validation else None
        self.num_samples_to_load = kcfg.get("num_train_subset")
        if self.num_samples_to_load is not None and self.num_samples_to_load <= 0:
            self.num_samples_to_load = None
        self.only_top1_gt = kcfg.get("only_top1_gt", True)
        
        self.crop_size = kcfg.get("crop_size", 0) 
        cropping_configs = kcfg.get("cropping_configs", {})
        
        # Determine method_weights based on validation status if not explicitly set for val
        default_train_weights = [0.2, 0.4, 0.4] # Contiguous, Spatial, SpatialInterface
        default_val_weights = [0.0, 0.0, 1.0]   # SpatialInterface only for validation (deterministic if seed is fixed)
        
        self.cropping_method_weights = cropping_configs.get("method_weights", 
                                                            default_val_weights if is_validation else default_train_weights)
        
        self.contiguous_crop_complete_lig = cropping_configs.get("contiguous_crop_complete_lig", False)
        self.spatial_crop_complete_lig = cropping_configs.get("spatial_crop_complete_lig", False)
        self.drop_last_contiguous = cropping_configs.get("drop_last", False)
        self.remove_metal_crop = cropping_configs.get("remove_metal", True)

        self.ref_pos_augment = kcfg.get("ref_pos_augment", True if not is_validation else False)
        self.lig_atom_rename_pf = kcfg.get("lig_atom_rename", False)

        df = pd.read_csv(labels_csv_path).fillna(-1e18)
        df["base_target_id"] = df["ID"].apply(lambda x: "_".join(x.split("_")[:-1]))
        df = df.sort_values(["base_target_id", "resid"])

        self.name_to_xyz: dict[str, np.ndarray] = {}
        self.name_to_seq_from_labels: dict[str, str] = {}

        for tid, grp in df.groupby("base_target_id"):
            seq_chars, xyz_per_res_list = [], []
            n_gt_max = (
                1 if self.only_top1_gt
                else max((int(c.split("_")[1]) for c in df.columns if c.startswith("x_")), default=1)
            )
            for _, row in grp.iterrows():
                seq_chars.append(row["resname"])
                coords_for_this_res = []
                for i in range(1, n_gt_max + 1):
                    x, y, z = row.get(f"x_{i}", -1e18), row.get(f"y_{i}", -1e18), row.get(f"z_{i}", -1e18)
                    if i == 1 and (x == -1e18 and y == -1e18 and z == -1e18):
                        coords_for_this_res = [[-1e18, -1e18, -1e18]] * n_gt_max
                        break
                    if x == -1e18 and y == -1e18 and z == -1e18:
                        break 
                    coords_for_this_res.append([x,y,z])
                while len(coords_for_this_res) < n_gt_max:
                    coords_for_this_res.append([-1e18, -1e18, -1e18])
                xyz_per_res_list.append(coords_for_this_res)
            
            if xyz_per_res_list:
                self.name_to_xyz[tid] = np.asarray(xyz_per_res_list, dtype=np.float32).transpose(1,0,2)
                self.name_to_seq_from_labels[tid] = "".join(seq_chars)

        seq_df = pd.read_csv(sequences_csv_path)
        
        # ** APPLY TEMPORAL CUTOFF FOR TRAINING SET **
        if not self.is_validation and self.apply_temporal_cutoff and self.temporal_cutoff_date:
            if 'temporal_cutoff' not in seq_df.columns:
                logger.error(f"'temporal_cutoff' column not found in {sequences_csv_path}. Cannot apply cutoff.")
            else:
                original_len = len(seq_df)
                # Ensure dates are in comparable format (pandas datetime objects)
                try:
                    seq_df['temporal_cutoff_dt'] = pd.to_datetime(seq_df['temporal_cutoff'], errors='coerce')
                    cutoff_date_dt = pd.to_datetime(self.temporal_cutoff_date)
                    
                    # Keep rows where temporal_cutoff is BEFORE the specified date
                    # Or where temporal_cutoff is NaT (missing), as we can't exclude them based on date
                    seq_df = seq_df[(seq_df['temporal_cutoff_dt'] < cutoff_date_dt) | (seq_df['temporal_cutoff_dt'].isna())]
                    
                    logger.info(f"Applied temporal cutoff: kept {len(seq_df)} out of {original_len} training samples (cutoff date < {self.temporal_cutoff_date}).")
                    if len(seq_df) == 0:
                        logger.warning("Temporal cutoff resulted in an empty training set!")
                except Exception as e:
                    logger.error(f"Error applying temporal cutoff: {e}. Proceeding with unfiltered training data.")
                    
        self.inputs: list[dict[str, Any]] = []
        processed = 0
        for _, row in seq_df.iterrows():
            if self.num_samples_to_load and processed >= self.num_samples_to_load:
                break
            
            tid = row["target_id"]
            canonical_sequence = str(row["sequence"])
            xyz_data_for_tid = self.name_to_xyz.get(tid)
            seq_from_labels = self.name_to_seq_from_labels.get(tid)

            if (xyz_data_for_tid is None or seq_from_labels is None or
                len(canonical_sequence) != xyz_data_for_tid.shape[1] or
                len(canonical_sequence) != len(seq_from_labels) or
                "-" in canonical_sequence or
                not np.any(xyz_data_for_tid[0, :, 0] > -1e17)):
                if DEBUG_FEATURIZE: logger.debug(f"Skipping target {tid} due to data inconsistency or invalid data.")
                continue
            
            self.inputs.append({
                "name": tid,
                "sequences": [{"rnaSequence": {"sequence": canonical_sequence, "count": 1}}],
                "original_full_sequence": canonical_sequence
            })
            processed += 1
        logger.info(f"Kaggle RNA3D | phase={'val' if self.is_validation else 'train'} | loaded {len(self.inputs)} samples.")

    def _featurise(self, sample_input_item: Mapping[str, Any]) -> Tuple[dict, dict, dict, str]:
        name = sample_input_item["name"]
        original_full_sequence_from_input = sample_input_item["original_full_sequence"]
        
        if DEBUG_FEATURIZE: logger.info(f"--- Starting _featurise for sample: {name}, L_input={len(original_full_sequence_from_input)} ---")

        xyz_all_gt_structures_full = self.name_to_xyz[name]
        xyz_top1_gt_full_from_input = xyz_all_gt_structures_full[0, :, :]

        # --- Stage 0: Emergency Pre-cropping for very long sequences ---
        # This sequence will be used for SDF and subsequent Protenix cropping.
        sequence_for_sdf = original_full_sequence_from_input
        xyz_for_sdf_gt = xyz_top1_gt_full_from_input

        if len(original_full_sequence_from_input) > PRE_CROP_THRESHOLD:
            if DEBUG_FEATURIZE: logger.warning(f"[{name}] L_input {len(original_full_sequence_from_input)} > PRE_CROP_THRESHOLD {PRE_CROP_THRESHOLD}. Applying linear pre-crop to {PRE_CROP_TARGET_LENGTH}.")
            
            # Use a simplified version of your old cropping logic
            # Ensure PRE_CROP_TARGET_LENGTH is feasible
            if PRE_CROP_TARGET_LENGTH >= len(original_full_sequence_from_input):
                 if DEBUG_FEATURIZE: logger.warning(f"[{name}] PRE_CROP_TARGET_LENGTH >= L_input. No pre-crop needed despite threshold.")
                 # sequence_for_sdf and xyz_for_sdf_gt remain as is
            else:
                valid_indices_orig = np.where(xyz_top1_gt_full_from_input[:, 0] > -1e17)[0]
                
                if not self.is_validation and len(valid_indices_orig) > 0: # Random pre-crop for training
                    min_valid_orig = valid_indices_orig.min()
                    max_valid_orig = valid_indices_orig.max()
                    
                    # Try to make the pre-crop window meaningful
                    low_bound = max(0, max_valid_orig - PRE_CROP_TARGET_LENGTH + 1)
                    high_bound = min(len(original_full_sequence_from_input) - PRE_CROP_TARGET_LENGTH, min_valid_orig)

                    if low_bound <= high_bound:
                        start_precrop_idx = np.random.randint(low_bound, high_bound + 1)
                    else: # Fallback if bounds are tricky (e.g. valid region too small or at ends)
                        max_start_offset = len(original_full_sequence_from_input) - PRE_CROP_TARGET_LENGTH
                        start_precrop_idx = np.random.randint(0, max_start_offset + 1) if max_start_offset >=0 else 0
                
                elif self.is_validation: # Deterministic pre-crop for validation (e.g. center)
                    start_precrop_idx = (len(original_full_sequence_from_input) - PRE_CROP_TARGET_LENGTH) // 2
                    start_precrop_idx = max(0, start_precrop_idx) # Ensure non-negative
                else: # No valid indices or validation with no clear center strategy, crop from start
                    start_precrop_idx = 0
                
                end_precrop_idx = start_precrop_idx + PRE_CROP_TARGET_LENGTH
                
                sequence_for_sdf = original_full_sequence_from_input[start_precrop_idx:end_precrop_idx]
                xyz_for_sdf_gt = xyz_top1_gt_full_from_input[start_precrop_idx:end_precrop_idx, :]
                if DEBUG_FEATURIZE: logger.info(f"[{name}] Pre-cropped to L={len(sequence_for_sdf)} (indices {start_precrop_idx}-{end_precrop_idx-1}).")

        # 1. Initial Featurization (on potentially pre-cropped sequence_for_sdf)
        if DEBUG_FEATURIZE: logger.info(f"[{name}] Step 1: SDF on sequence of length {len(sequence_for_sdf)}...")
        sdf_input_sample = {"name": name, "sequences": [{"rnaSequence": {"sequence": sequence_for_sdf, "count": 1}}]}
        try:
            sdf_instance = SampleDictToFeatures(sdf_input_sample)
            # atom_arr_intermediate and token_arr_intermediate now correspond to sequence_for_sdf
            _, atom_arr_intermediate, token_arr_intermediate = sdf_instance.get_feature_dict()
        except Exception as e:
            logger.error(f"SampleDictToFeatures failed for {name} (L={len(sequence_for_sdf)}): {e}")
            raise 
        if DEBUG_FEATURIZE: logger.info(f"[{name}] atom_arr_intermediate: {len(atom_arr_intermediate)} atoms, token_arr_intermediate: {len(token_arr_intermediate)} tokens.")

        # --- Add essential annotations to atom_arr_intermediate ---
        is_resolved_atom_mask_intermediate = np.zeros(len(atom_arr_intermediate), dtype=bool)
        for token_idx_inter, token_obj_inter in enumerate(token_arr_intermediate):
            if token_idx_inter < len(xyz_for_sdf_gt) and xyz_for_sdf_gt[token_idx_inter, 0] > -1e17:
                is_resolved_atom_mask_intermediate[token_obj_inter.atom_indices] = True
        if not hasattr(atom_arr_intermediate, 'is_resolved'): atom_arr_intermediate.set_annotation("is_resolved", is_resolved_atom_mask_intermediate)
        else: atom_arr_intermediate.is_resolved = is_resolved_atom_mask_intermediate
        
        # Add other annotations needed by CropData/MSAFeaturizer to atom_arr_intermediate
        if not hasattr(atom_arr_intermediate, 'label_entity_id'):
            atom_arr_intermediate.set_annotation("label_entity_id", np.array(["1"] * len(atom_arr_intermediate), dtype='<U8'))
        if not hasattr(atom_arr_intermediate, 'asym_id_int'):
            atom_arr_intermediate.set_annotation("asym_id_int", np.zeros(len(atom_arr_intermediate), dtype=int))
        if not hasattr(atom_arr_intermediate, 'mol_type'):
            atom_arr_intermediate.set_annotation("mol_type", np.array(["rna"] * len(atom_arr_intermediate), dtype='<U7'))
        if not hasattr(atom_arr_intermediate, 'centre_atom_mask'):
            centers = np.zeros(len(atom_arr_intermediate), dtype=bool)
            for tf_inter in token_arr_intermediate: centers[tf_inter.centre_atom_index] = True
            atom_arr_intermediate.set_annotation("centre_atom_mask", centers)
        if not hasattr(atom_arr_intermediate, 'ref_space_uid'):
            res_ids_inter = atom_arr_intermediate.res_id 
            unique_res_ids_inter, u_indices_inter = np.unique(res_ids_inter, return_inverse=True)
            atom_arr_intermediate.set_annotation("ref_space_uid", u_indices_inter.astype(int))
        if DEBUG_FEATURIZE: logger.info(f"[{name}] Added basic annotations to atom_arr_intermediate.")

        # --- Protenix-style Cropping (operates on atom_arr_intermediate, token_arr_intermediate) ---
        selected_token_indices_np = np.arange(len(token_arr_intermediate)) # Default (indices into token_arr_intermediate)
        protenix_crop_successful = False

        if self.crop_size > 0 and len(token_arr_intermediate) > self.crop_size:
            if DEBUG_FEATURIZE: logger.info(f"[{name}] Step 2: Attempting Protenix Cropping (L_intermed={len(token_arr_intermediate)}, crop_to={self.crop_size})...")
            ref_chain_asym_id_int = np.unique(atom_arr_intermediate.asym_id_int)[0] if hasattr(atom_arr_intermediate, 'asym_id_int') and len(np.unique(atom_arr_intermediate.asym_id_int)) > 0 else 0
            
            crop_instance = CropData(
                crop_size=self.crop_size, ref_chain_indices=[ref_chain_asym_id_int],
                token_array=token_arr_intermediate, atom_array=atom_arr_intermediate, 
                method_weights=self.cropping_method_weights,
                contiguous_crop_complete_lig=self.contiguous_crop_complete_lig,
                spatial_crop_complete_lig=self.spatial_crop_complete_lig,
                drop_last=self.drop_last_contiguous, remove_metal=self.remove_metal_crop
            )
            chosen_method = crop_instance.random_crop_method()
            if DEBUG_FEATURIZE: logger.info(f"[{name}] Chosen crop method: {chosen_method}")
            try:
                selected_token_indices_tensor, _ = crop_instance.get_crop_indices(crop_method=chosen_method)
                selected_token_indices_np = selected_token_indices_tensor.cpu().numpy() # These are indices into token_arr_intermediate
                if not (0 < len(selected_token_indices_np) <= self.crop_size):
                    raise ValueError(f"Protenix crop returned invalid token count: {len(selected_token_indices_np)}")
                protenix_crop_successful = True
                if DEBUG_FEATURIZE: logger.info(f"[{name}] Protenix crop successful, {len(selected_token_indices_np)} tokens selected from intermediate.")
            except Exception as e:
                logger.error(f"[{name}] Protenix CropData for (L_intermed={len(token_arr_intermediate)}) (method {chosen_method}) failed: {e}. Attempting fallback linear crop on intermediate.")
                protenix_crop_successful = False
        else:
            if DEBUG_FEATURIZE: logger.info(f"[{name}] No Protenix crop needed (L_intermed <= crop_size or crop_size=0). Using {len(selected_token_indices_np)} tokens from intermediate.")

        if not protenix_crop_successful and self.crop_size > 0 and len(token_arr_intermediate) > self.crop_size:
            if DEBUG_FEATURIZE: logger.info(f"[{name}] Applying fallback linear crop on intermediate (L_intermed={len(token_arr_intermediate)}, crop_to={self.crop_size})")
            # Linear crop on token_arr_intermediate
            valid_c1_indices_inter = [idx for idx, t in enumerate(token_arr_intermediate) if idx < len(xyz_for_sdf_gt) and xyz_for_sdf_gt[idx, 0] > -1e17]
            start_idx = 0
            if not valid_c1_indices_inter: 
                if DEBUG_FEATURIZE: logger.warning(f"[{name}] No valid C1's for fallback on intermediate. Cropping from start.")
            else:
                min_valid_idx, max_valid_idx = min(valid_c1_indices_inter), max(valid_c1_indices_inter)
                max_start_offset = len(token_arr_intermediate) - self.crop_size
                low_bound = max(0, max_valid_idx - self.crop_size + 1)
                high_bound = min(max_start_offset, min_valid_idx)
                if low_bound <= high_bound: start_idx = np.random.randint(low_bound, high_bound + 1)
                elif min_valid_idx < self.crop_size: start_idx = 0
                elif (len(token_arr_intermediate) - max_valid_idx - 1) < self.crop_size: start_idx = max_start_offset
                else: start_idx = np.random.randint(0, max_start_offset + 1) if max_start_offset >=0 else 0
            
            end_idx = start_idx + self.crop_size
            selected_token_indices_np = np.arange(start_idx, end_idx) # Indices into token_arr_intermediate
            if DEBUG_FEATURIZE: logger.info(f"[{name}] Fallback linear crop on intermediate: indices {start_idx}-{end_idx-1}.")

        if len(selected_token_indices_np) > self.crop_size and self.crop_size > 0:
            if DEBUG_FEATURIZE: logger.warning(f"[{name}] Final selected_token_indices_np for intermediate array too long ({len(selected_token_indices_np)} > {self.crop_size}). Truncating.")
            selected_token_indices_np = selected_token_indices_np[:self.crop_size]
        
        # --- MSA Featurization ---
        if DEBUG_FEATURIZE: logger.info(f"[{name}] Step 3: MSA Featurization...")
        msa_features_aligned_to_intermediate = None
        msa_features_added_flag = False
        if self.msa_featurizer:
            bioassembly_for_msa = {"pdb_id": name, "sequences": {}, "atom_array": atom_arr_intermediate,
                                   "token_array": token_arr_intermediate, "entity_poly_type": {}}
            current_entity_ids_inter = np.unique(atom_arr_intermediate.label_entity_id)
            rna_entity_id_str_inter = str(current_entity_ids_inter[0]) if len(current_entity_ids_inter) > 0 else "1"
            bioassembly_for_msa["sequences"][rna_entity_id_str_inter] = original_full_sequence_from_input # True original for lookup
            bioassembly_for_msa["entity_poly_type"][rna_entity_id_str_inter] = "polyribonucleotide"
            entity_to_asym_id_int_map_inter = {}
            asym_ids_inter = np.unique(atom_arr_intermediate.asym_id_int[atom_arr_intermediate.label_entity_id == rna_entity_id_str_inter])
            entity_to_asym_id_int_map_inter[rna_entity_id_str_inter] = asym_ids_inter.tolist() if len(asym_ids_inter) > 0 else [0]

            try:
                msa_features_aligned_to_intermediate = self.msa_featurizer(
                    bioassembly_dict=bioassembly_for_msa,
                    selected_indices=selected_token_indices_np, # These are indices for token_arr_intermediate
                    entity_to_asym_id_int=entity_to_asym_id_int_map_inter)
                if msa_features_aligned_to_intermediate and len(msa_features_aligned_to_intermediate) > 0:
                    msa_features_added_flag = True
                    if DEBUG_FEATURIZE: logger.info(f"[{name}] MSA features obtained (aligned to intermediate). Example shape (msa): {msa_features_aligned_to_intermediate.get('msa', np.array([])).shape}")
            except Exception as msa_exc:
                logger.error(f"[{name}] MSA featurization error: {msa_exc}")
                if DEBUG_FEATURIZE: traceback.print_exc()
        elif DEBUG_FEATURIZE: logger.info(f"[{name}] MSA featurizer is None.")

        # 4. Final Cropping of Atom/Token Arrays AND MSA features based on selected_token_indices_np
        if DEBUG_FEATURIZE: logger.info(f"[{name}] Step 4: Final Cropping of arrays/features (select_by_token_indices)...")
        (cropped_token_array, cropped_atom_array, 
         cropped_msa_features_np, _) = CropData.select_by_token_indices(
            token_array=token_arr_intermediate, atom_array=atom_arr_intermediate,
            selected_token_indices=torch.from_numpy(selected_token_indices_np).long(),
            msa_features=msa_features_aligned_to_intermediate, template_features=None)
        if DEBUG_FEATURIZE: logger.info(f"[{name}] Final crop: {len(cropped_token_array)} tokens, {len(cropped_atom_array)} atoms.")
        if DEBUG_FEATURIZE and cropped_msa_features_np and 'msa' in cropped_msa_features_np:
            logger.info(f"[{name}] Final cropped MSA shape: {cropped_msa_features_np['msa'].shape}")


        # 5. Generate Final `feat_dict` using ProtenixFeaturizer on CROPPED data
        if DEBUG_FEATURIZE: logger.info(f"[{name}] Step 5: ProtenixFeaturizer on final cropped data...")
        final_protenix_featurizer = ProtenixFeaturizer(
            cropped_token_array=cropped_token_array, cropped_atom_array=cropped_atom_array,
            ref_pos_augment=self.ref_pos_augment if not self.is_validation else False,
            lig_atom_rename=self.lig_atom_rename_pf)
        final_feat_dict = final_protenix_featurizer.get_all_input_features()
        if hasattr(final_protenix_featurizer, 'get_atom_permutation_list'):
             final_feat_dict["atom_perm_list"] = final_protenix_featurizer.get_atom_permutation_list()
        if DEBUG_FEATURIZE: logger.info(f"[{name}] final_feat_dict created. Num keys: {len(final_feat_dict)}")

        if cropped_msa_features_np and len(cropped_msa_features_np) > 0:
            final_feat_dict.update(dict_to_tensor(cropped_msa_features_np))
        else: msa_features_added_flag = False 

        # 6. Ground Truth Labeling for Cropped Data
        if DEBUG_FEATURIZE: logger.info(f"[{name}] Step 6: GT Labeling...")
        # GT coords must be indexed based on selected_token_indices_np relative to xyz_for_sdf_gt
        gt_coords_for_final_selection = xyz_for_sdf_gt[selected_token_indices_np, :]
        
        final_gt_coords_np = np.full((len(cropped_atom_array), 3), 0.0, dtype=np.float32)
        final_gt_mask_np = np.full(len(cropped_atom_array), 0, dtype=np.int64)
        for i, token_obj in enumerate(cropped_token_array): # Iterate through the *final* cropped tokens
            c1_atom_idx_in_final_cropped = token_obj.centre_atom_index
            gt_coord_for_this_final_token = gt_coords_for_final_selection[i] # Match by index
            if gt_coord_for_this_final_token[0] > -1e17:
                final_gt_coords_np[c1_atom_idx_in_final_cropped] = gt_coord_for_this_final_token
                final_gt_mask_np[c1_atom_idx_in_final_cropped] = 1
        labels = {"coordinate": torch.from_numpy(final_gt_coords_np),
                  "coordinate_mask": torch.from_numpy(final_gt_mask_np)}

        # 7. Dummy Features & Data Type Transform
        if DEBUG_FEATURIZE: logger.info(f"[{name}] Step 7: Dummy Features & Type Transform...")
        dummy_feature_list = ["template"]
        if not msa_features_added_flag:
            dummy_feature_list.append("msa")
        final_feat_dict = make_dummy_feature(final_feat_dict, dummy_feature_list)
        final_feat_dict = data_type_transform(final_feat_dict)

        # 8. label_full_dict
        if DEBUG_FEATURIZE: logger.info(f"[{name}] Step 8: label_full_dict construction...")
        label_full = {**final_feat_dict, **labels}

        if labels["coordinate_mask"].sum() == 0:
            log_msg = f"Sample {name}: coordinate_mask sum is 0 after all processing."
            if not self.is_validation: logger.warning(log_msg + " This might be problematic for training.")
            else: logger.warning(log_msg)
        
        if DEBUG_FEATURIZE: logger.info(f"--- Finished _featurise for {name} ---")
        return final_feat_dict, labels, label_full, name
    
    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.inputs[idx]
        try:
            feat, lbl, lbl_full, name = self._featurise(s)
            
            basic_info = {
                "pdb_id": name,
                "N_token": torch.tensor([feat["token_index"].shape[0]]),
                "N_atom": torch.tensor([feat["atom_to_token_idx"].shape[0]]),
            }
            if "msa" in feat and isinstance(feat["msa"], torch.Tensor) and feat["msa"].ndim >=2 :
                basic_info["N_msa"] = torch.tensor([feat["msa"].shape[0]])
                for msa_count_key in ["prot_pair_num_alignments", "prot_unpair_num_alignments",
                                      "rna_pair_num_alignments", "rna_unpair_num_alignments"]:
                    # These keys are added by MSAFeaturizer if it runs successfully.
                    # Featurizer (ProtenixFeaturizer) does not add them.
                    # MakeDummyFeature adds them if "msa" is in dummy_feats.
                    # So, they should generally be present if MSA pipeline (real or dummy) ran.
                    basic_info[msa_count_key] = feat.get(msa_count_key, torch.tensor(0, dtype=torch.int32))
            else:
                basic_info["N_msa"] = torch.tensor(0) # No MSA feature tensor
                for msa_count_key in ["prot_pair_num_alignments", "prot_unpair_num_alignments",
                                      "rna_pair_num_alignments", "rna_unpair_num_alignments"]:
                     basic_info[msa_count_key] = torch.tensor(0, dtype=torch.int32)
            return {
                "input_feature_dict": feat,
                "label_dict":         lbl,
                "label_full_dict":    lbl_full,
                "basic":              basic_info,
            }
        except Exception as e:
            logger.error(f"Error in __getitem__ for sample {s.get('name', 'UNKNOWN')} (index {idx}): {e}")
            traceback.print_exc()
            logger.warning(f"Retrying with next sample: {(idx + 1) % len(self)}")
            return self.__getitem__((idx + 1) % len(self))

# --------------------------------------------------------------------------- #
# 2.  Thin wrapper that the factory looks for
# --------------------------------------------------------------------------- #
class KaggleRNADataset(Dataset):
    def __init__(self, cfg: Mapping[str, Any], phase: str, msa_featurizer_instance: Optional[MSAFeaturizer] = None):
        assert phase in ("train", "val")
       
        self._impl = _SimpleRNADataset(
            kcfg=cfg, # Pass the specific kaggle_rna3d config part
            sequences_csv_path=os.path.join(cfg["root"],
                cfg["val_sequences_csv" if phase == "val" else "train_sequences_csv"]),
            labels_csv_path=os.path.join(cfg["root"],
                cfg["val_labels_csv"    if phase == "val" else "train_labels_csv"]),
            is_validation     = (phase == "val"),
            msa_featurizer    = msa_featurizer_instance,
            apply_temporal_cutoff = (phase == "train" and APPLY_TEMPORAL_CUTOFF_FOR_TRAIN),
            temporal_cutoff_date  = TEMPORAL_CUTOFF_DATE_STR if (phase == "train") else None
        )
    def __len__(self): return len(self._impl)
    def __getitem__(self, i): return self._impl[i]