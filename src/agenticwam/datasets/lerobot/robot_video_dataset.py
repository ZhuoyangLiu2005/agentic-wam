import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from agenticwam.utils.logging_config import get_logger
from agenticwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        context_len=128,
        cosmos_text_cache_dir=None,   # dir of cached Cosmos-Reason1 prompt_embs; set -> sample['prompt_embs']
        episode_instruction_map=None, # path to {episode_index: primary_instruction} json (per-episode text cond)
        code_instruction_mode: str = "shared",  # 'shared' (default, unchanged) | 'episode'
        skill_prompt_map=None,  # EXP-1: path to {canonical_instruction: skill_block}. None -> off.
        skill_prompt_only: bool = False,  # code expert sees ONLY the skill block (no instruction):
        #   removes the language->code shortcut (verified harmful in pretraining); the target can
        #   reach code only via the video expert, which stays instruction-conditioned (prompt_embs).
        # When set, the CODE expert's prompt is prefixed with the task-relevant skill defs
        # (retrieval, ~65-250 tok) so it can copy exact signatures instead of recalling them.
        # 'episode': the CODE expert gets this episode's OWN (randomised) instruction while the
        # video/action experts keep the canonical primary (so the Cosmos embed cache stays small).
        # Emitted as a separate `prompt_code` field; absent under 'shared' -> callers fall back.
        instruction_idx_path=None, # flat frame-indexed instruction cache (CORRECT); episode-map lookup was buggy (local ep_idx into global map)
        frame_offset=0,  # global-frame offset: local val idx -> global suffix cache idx (suffix val split)
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        use_policy_code: bool = False,  # attach sample["policy_code"] for the tri-modal coding expert
        generate_template_policy_code: bool = False,  # synthesize template code from the instruction
        policy_code_cache_dir: Optional[str] = None,  # dir of GT code keyed by sha256(raw task); template fallback
        policy_code_coverage: float = 1.0,  # fraction of samples (by instruction hash) that get GT code supervision; rest -> "" (masked)
        episode_indices: Optional[list] = None,  # explicit episode subset (overrides the shuffle split)
        skill_labels_path: Optional[str] = None,  # flat per-sample skill_id cache (Step 4 skill conditioning)
        obs_grounded_code: bool = False,  # emit per-chunk function-like code from the obs-grounded skill
        code_slices: bool = False,  # CodeStream: per-cycle slice as policy_code + history slices in prompt
        code_slices_cache_dir: Optional[str] = None,  # dir of {sha256(task): {"slices": [...]}}
        policy_code_idx_path: Optional[str] = None,  # flat {idx->code} obs-grounded per-episode cache
        cycle_cache_path: Optional[str] = None,  # idx->cycle (which object) cache .pt
        code_slices_no_history: bool = False,  # ablation: omit completed-slice history from prompt
        code_slices_summary: bool = False,  # BOUNDED event log: "# Completed: k/n. Last: <slice>" (vs full transcript)
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            episode_indices=(episode_indices if isinstance(episode_indices, (str, dict))  # YAMFIX-PERSUB2
                            else list(episode_indices)) if episode_indices is not None else None,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.context_len = context_len
        self.cosmos_text_cache_dir = cosmos_text_cache_dir
        self.code_instruction_mode = str(code_instruction_mode or "shared")
        self.skill_prompt_only = bool(skill_prompt_only)
        self.skill_prompt_map = None
        if skill_prompt_map is not None:
            import json as _js
            self.skill_prompt_map = _js.load(open(skill_prompt_map))
            logger.info("Loaded skill-prompt map: %s (%d tasks)", skill_prompt_map, len(self.skill_prompt_map))
        self.episode_instruction_map = None
        if episode_instruction_map is not None:
            import json as _json
            self.episode_instruction_map = _json.load(open(episode_instruction_map))
            logger.info("Loaded per-episode instruction map: %s (%d episodes)",
                        episode_instruction_map, len(self.episode_instruction_map))
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.frame_offset = int(frame_offset)
        self.instructions = None; self.instruction_id_by_idx = None
        if instruction_idx_path is not None:
            import torch as _torch
            _im = _torch.load(instruction_idx_path, map_location="cpu")
            self.instructions = _im["instructions"]; self.instruction_id_by_idx = _im["instruction_id_by_idx"]
            logger.info("Loaded flat instruction cache: %s (%d instr, %d frames)", instruction_idx_path, len(self.instructions), len(self.instruction_id_by_idx))
        self.use_policy_code = use_policy_code
        self.generate_template_policy_code = generate_template_policy_code
        self.policy_code_cache_dir = policy_code_cache_dir
        self.policy_code_coverage = float(policy_code_coverage)
        # Step 4: per-sample obs-grounded skill ids (indexed by lerobot sample idx)
        self.skill_by_idx = None
        self.obs_grounded_code = bool(obs_grounded_code)
        self.code_slices = bool(code_slices)
        self.code_slices_no_history = bool(code_slices_no_history)
        self.code_slices_summary = bool(code_slices_summary)
        self.code_slices_cache_dir = code_slices_cache_dir
        self.codes = None
        self.code_id_by_idx = None
        if policy_code_idx_path:
            import torch as _t
            _d = _t.load(policy_code_idx_path)
            self.codes = _d["codes"]; self.code_id_by_idx = _d["code_id_by_idx"]
            logger.info("Loaded obs-grounded code cache: %s (%d codes, %d idx)",
                        policy_code_idx_path, len(self.codes), len(self.code_id_by_idx))
        self.cycle_by_idx = None
        if cycle_cache_path is not None:
            import torch as _t
            self.cycle_by_idx = _t.load(cycle_cache_path)["cycle_by_idx"]
            logger.info("Loaded cycle cache: %s (%d samples) code_slices=%s", cycle_cache_path, len(self.cycle_by_idx), self.code_slices)
        if skill_labels_path:
            import torch as _torch
            self.skill_by_idx = _torch.load(skill_labels_path)["skill_by_idx"]
            logger.info("Loaded skill labels: %s (%d samples) obs_grounded_code=%s",
                        skill_labels_path, len(self.skill_by_idx), self.obs_grounded_code)

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "multiview":
            # Cosmos-native multiview: keep views SEPARATE, stacked along the TIME axis
            # -> [nc*T_video, C, H, W]. The model re-splits per view for per-view VAE encode,
            # view embeddings and multiview cross-attn. (Do NOT spatially composite.)
            video = video.reshape(num_cameras * T_video, C, H, W)
            self._n_views = num_cameras
        elif self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        # For multiview (views stacked on the time axis), the action aligns with a SINGLE view's
        # timeline, so check against per-view transitions (frames/view - 1), not the total.
        _nv = getattr(self, "_n_views", 1) if self.concat_multi_camera == "multiview" else 1
        _vt_per_view = video.shape[1] // _nv
        if action.shape[0] % (_vt_per_view - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by per-view `video` transitions, "
                f"got {action.shape[0]} and {_vt_per_view - 1} (n_views={_nv})"
            )

        task = sample["instruction"]
        if self.instruction_id_by_idx is not None:
            task = self.instructions[int(self.instruction_id_by_idx[self.frame_offset + int(sample["idx"])])]
        elif self.episode_instruction_map is not None:
            # Per-episode text conditioning: use the episode's primary instruction (one per
            # trajectory) so the Cosmos-Reason1 embed cache is small (unique primaries only).
            ei = sample.get("episode_index", None)
            if ei is not None and str(ei) in self.episode_instruction_map:
                task = self.episode_instruction_map[str(ei)]
        elif self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        # Dual-instruction: the code expert may use this episode's OWN instruction (raw, randomised)
        # rather than the canonical primary. Same DEFAULT_PROMPT wrapper so the prefix format the code
        # expert was trained on is preserved.
        prompt_code = None
        if self.skill_prompt_map is not None:
            # ALWAYS emit prompt_code when the map is active, even when the block is empty
            # (shake_bottle uses raw primitives -> 0 skills). Emitting it conditionally makes the
            # key present in some samples and absent in others -> default collate raises
            # KeyError: 'prompt_code' in the DataLoader worker.
            prompt_code = self.skill_prompt_map.get(task, "")
            if not self.skill_prompt_only:
                prompt_code = prompt_code + instruction
        if self.code_instruction_mode == "episode":
            _own = sample.get("instruction", None)
            if isinstance(_own, str) and _own.strip():
                prompt_code = DEFAULT_PROMPT.format(task=_own)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            **({"prompt_code": prompt_code} if prompt_code is not None else {}),
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.cosmos_text_cache_dir is not None:
            # Cosmos backbone: precomputed Cosmos-Reason1 embeds; no umT5 context needed.
            data["prompt_embs"] = self._get_cached_cosmos_embs(instruction)
        else:
            raise ValueError("cosmos_text_cache_dir is not set: precompute the instruction embeddings with "
                             "scripts/precompute_cosmos_text_embeds.py (docs/DATA.md section 4)")
        if self.use_policy_code:
            data["policy_code"] = self._get_policy_code(task, sample)
        if self.skill_by_idx is not None:
            _sk = int(self.skill_by_idx[self.frame_offset + int(sample["idx"])])
            if self.obs_grounded_code:
                from agenticwam.coding.skill_code import skill_to_policy_code
                data["policy_code"] = skill_to_policy_code(_sk, task)
            else:
                data["skill_id"] = _sk
        if self.code_id_by_idx is not None:
            data["policy_code"] = self.codes[int(self.code_id_by_idx[self.frame_offset + int(sample["idx"])])]
        if self.code_slices and self.cycle_by_idx is not None:
            # [dataprep] the episode index selects the per-episode slice file (see _get_code_slice)
            cur, hist_prompt, hist_txt, prompt_nh = self._get_code_slice(
                task, self.frame_offset + int(sample["idx"]), sample.get("episode_index", None))
            data["policy_code"] = cur
            data["prompt"] = hist_prompt
            data["prompt_code"] = hist_prompt   # code expert prefers prompt_code; without this the
                                                # skill-only block wins and the event log vanishes
            data["code_history"] = hist_txt
            data["prompt_nohist"] = prompt_nh
        return data

    def _get_code_slice(self, task: str, idx: int, episode_index=None):
        """CodeStream: return (current_slice, prompt_with_history) for the chunk's cycle.
        prompt = instruction + completed slices (masked in build_code_batch); current slice is supervised.

        [dataprep] Slice lookup order:
          1. per-EPISODE file `ep_%06d.json` (exact-alignment caches written by
             robotwin/dataprep/slicing.py — two episodes of one instruction can
             differ in their cycles, which an instruction-keyed file cannot express)
          2. per-instruction file `sha256(task).json` (the original heuristic caches)
        A `preamble` key in the file (the program's helper library) is prompt-side conditioning,
        inserted right after the instruction and never supervised."""
        import json, hashlib
        base = "# Instruction: %s\n" % task
        if self.code_slices_cache_dir is None or self.cycle_by_idx is None:
            return "", base
        cache_path = None   # [dataprep] per-episode file first, instruction-keyed file as fallback
        if episode_index is not None:
            ep_path = os.path.join(self.code_slices_cache_dir, "ep_%06d.json" % int(episode_index))
            if os.path.exists(ep_path):
                cache_path = ep_path
        if cache_path is None:
            hashed = hashlib.sha256(task.encode("utf-8")).hexdigest()
            cache_path = os.path.join(self.code_slices_cache_dir, f"{hashed}.json")
        slices, preamble = [], ""
        if os.path.exists(cache_path):
            try:
                blob = json.loads(open(cache_path).read())
                slices = blob.get("slices", [])
                preamble = blob.get("preamble", "")   # [dataprep] helper library; absent in older caches
            except Exception:
                slices = []
        if not slices:
            return "", base, "", base
        cyc = int(self.cycle_by_idx[idx])
        cyc = max(0, min(cyc, len(slices) - 1))
        current = slices[cyc]
        lib = ("# Helper library:\n%s" % preamble) if preamble else ""   # [dataprep]
        history = "" if getattr(self, "code_slices_no_history", False) else "".join(slices[:cyc])
        # prefix: skill block (skill-only conditioning) when the map is active, else the instruction
        _skill = ""
        if getattr(self, "skill_prompt_map", None) is not None and getattr(self, "skill_prompt_only", False):
            _skill = self.skill_prompt_map.get(task, "")
            head = _skill
        else:
            head = "# Instruction: %s\n" % task
        head = head + lib   # [dataprep] helper library right after the instruction (empty when unset)
        if getattr(self, "code_slices_no_history", False):
            prompt = head + "# Next:\n"
        elif getattr(self, "code_slices_summary", False):
            # BOUNDED event log: count + last slice verbatim. The verbatim-history variant measured
            # 1,098 tokens worst-case (put_bottles) vs the 512-position cap -- the transcript form
            # silently strangles the generation canvas (gen_len = cap - plen). Constant-size instead.
            _last = slices[cyc - 1] if cyc > 0 else ""
            if _last and not _last.endswith("\n"):
                _last += "\n"          # deploy must compose byte-identically; keep the boundary clean
            prompt = head + ("# Completed: %d/%d cycles. Last:\n%s# Next:\n" % (cyc, len(slices), _last))
        else:
            prompt = head + ("# Completed:\n%s# Next:\n" % history)
        # BOTH-design: with code_memory on, training consumes prompt_nohist -- in summary mode that
        # must BE the bounded-log prompt so the event log and the memory slots co-train (the user's
        # short+long design), not a logless variant that silently drops the A2 component.
        _nh = prompt if getattr(self, "code_slices_summary", False) else ("# Instruction: %s\n%s# Next:\n" % (task, lib))   # [dataprep] lib
        return current, prompt, history, _nh

    def _get_policy_code(self, task: str, sample: dict) -> str:
        """Resolve the GT policy code for an instruction. Priority: on-disk GT cache
        (keyed by sha256(raw task); its generator was removed) ->
        template synthesis -> any upstream `policy_code` field -> empty string."""
        # Code-coverage ablation: deterministically (by instruction hash) withhold GT
        # code from a (1 - coverage) fraction of instructions. Withheld -> "" which
        # build_code_batch fully masks (no code loss), but the code stream still runs on
        # the instruction, so only the SUPERVISION density changes (DDP-safe: the
        # trainable joint-attn params still participate in every forward).
        if self.policy_code_coverage < 1.0:
            bucket = int(hashlib.sha256(("cov::" + task).encode("utf-8")).hexdigest(), 16) % 10000
            if bucket >= int(self.policy_code_coverage * 10000):
                return ""
        if self.policy_code_cache_dir is not None:
            hashed = hashlib.sha256(task.encode("utf-8")).hexdigest()
            cache_path = os.path.join(self.policy_code_cache_dir, f"{hashed}.json")
            if os.path.exists(cache_path):
                try:
                    import json
                    code = json.loads(open(cache_path).read()).get("policy_code", "")
                    if code:
                        return code
                except Exception:
                    pass  # fall through to template
        if self.generate_template_policy_code or self.policy_code_cache_dir is not None:
            from agenticwam.coding import generate_template_policy_code
            return generate_template_policy_code(task)
        return sample.get("policy_code", "")

    def _get_cached_cosmos_embs(self, prompt: str):
        """Load precomputed Cosmos-Reason1 embeds [N, 100352] for `prompt`
        (written by scripts/precompute_cosmos_text_embeds.py, keyed by sha256(prompt))."""
        from .prompt_embs_store import get as _pe_get
        _e = _pe_get(prompt)                 # consolidated node-local mmap (see prompt_embs_store)
        if _e is not None:                   # 0.02 ms vs 36 ms + 25.7 MB Lustre read per sample
            return _e
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cosmos_text_cache_dir, f"{hashed}.cosmos_reason1.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing Cosmos text embed cache: {cache_path}. "
                "Run scripts/precompute_cosmos_text_embeds.py first."
            )
        return torch.load(cache_path, map_location="cpu")["prompt_embs"]

    def __getitem__(self, idx):
        for _attempt in range(25):
            _use = idx if _attempt == 0 else int(np.random.randint(len(self)))
            try:
                return self._get(_use)
            except Exception as e:
                if _attempt == 0:
                    print(f"Error processing sample idx {idx}: {e}. Retrying with random samples.")
                if _attempt >= 24:
                    raise
        raise RuntimeError("unreachable")
