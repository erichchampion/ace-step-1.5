"""Training run wiring helpers extracted from ``events.__init__``."""

from typing import Any, Iterator

from loguru import logger

from .. import training_handlers as train_h
from .context import TrainingWiringContext

def _normalize_training_state(training_state: Any) -> dict[str, bool]:
    """Return a valid mutable training-state mapping for streaming wrappers."""

    if isinstance(training_state, dict):
        return training_state
    return {"is_training": False, "should_stop": False}


def _build_training_wrapper(dit_handler: Any):
    """Build the training stream wrapper bound to the current DiT handler."""

    def training_wrapper(
        tensor_dir: Any,
        r: Any,
        a: Any,
        d: Any,
        lr: Any,
        ep: Any,
        bs: Any,
        ga: Any,
        se: Any,
        sh: Any,
        sd: Any,
        od: Any,
        rc: Any,
        ts: Any,
    ) -> Iterator[tuple[Any, Any, Any, dict[str, bool]]]:
        """Stream LoRA training progress and normalize failure outputs for UI."""

        state = _normalize_training_state(ts)
        try:
            for progress, log_msg, plot, next_state in train_h.start_training(
                tensor_dir, dit_handler, r, a, d, lr, ep, bs, ga, se, sh, sd, od, rc, state
            ):
                yield progress, log_msg, plot, next_state
        except Exception as exc:  # pragma: no cover - defensive UI wrapper
            logger.exception("Training wrapper error")
            yield f"❌ Error: {str(exc)}", str(exc), None, state

    return training_wrapper


def _build_lokr_training_wrapper(dit_handler: Any):
    """Build the LoKr training stream wrapper bound to the current DiT handler."""

    def lokr_training_wrapper(
        tensor_dir: Any,
        ldim: Any,
        lalpha: Any,
        factor: Any,
        decompose_both: Any,
        use_tucker: Any,
        use_scalar: Any,
        weight_decompose: Any,
        lr: Any,
        ep: Any,
        bs: Any,
        ga: Any,
        se: Any,
        sh: Any,
        sd: Any,
        od: Any,
        ts: Any,
    ) -> Iterator[tuple[Any, Any, Any, dict[str, bool]]]:
        """Stream LoKr training progress and normalize failure outputs for UI."""

        state = _normalize_training_state(ts)
        try:
            for progress, log_msg, plot, next_state in train_h.start_lokr_training(
                tensor_dir,
                dit_handler,
                ldim,
                lalpha,
                factor,
                decompose_both,
                use_tucker,
                use_scalar,
                weight_decompose,
                lr,
                ep,
                bs,
                ga,
                se,
                sh,
                sd,
                od,
                state,
            ):
                yield progress, log_msg, plot, next_state
        except Exception as exc:  # pragma: no cover - defensive UI wrapper
            logger.exception("LoKr training wrapper error")
            yield f"❌ Error: {str(exc)}", str(exc), None, state

    return lokr_training_wrapper


def register_training_run_handlers(context: TrainingWiringContext) -> None:
    """Register training/LoKr run-tab handlers with stable IO ordering."""

    training_section = context.training_section
    training_wrapper = _build_training_wrapper(context.dit_handler)
    lokr_training_wrapper = _build_lokr_training_wrapper(context.dit_handler)

    # ========== Training Tab Handlers ==========
    training_section["load_dataset_btn"].click(
        fn=train_h.load_training_dataset,
        inputs=[training_section["training_tensor_dir"]],
        outputs=[training_section["training_dataset_info"]],
    )

    training_section["start_training_btn"].click(
        fn=training_wrapper,
        inputs=[
            training_section["training_tensor_dir"],
            training_section["lora_rank"],
            training_section["lora_alpha"],
            training_section["lora_dropout"],
            training_section["learning_rate"],
            training_section["train_epochs"],
            training_section["train_batch_size"],
            training_section["gradient_accumulation"],
            training_section["save_every_n_epochs"],
            training_section["training_shift"],
            training_section["training_seed"],
            training_section["lora_output_dir"],
            training_section["resume_checkpoint_dir"],
            training_section["training_state"],
        ],
        outputs=[
            training_section["training_progress"],
            training_section["training_log"],
            training_section["training_loss_plot"],
            training_section["training_state"],
        ],
    )

    training_section["stop_training_btn"].click(
        fn=train_h.stop_training,
        inputs=[training_section["training_state"]],
        outputs=[
            training_section["training_progress"],
            training_section["training_state"],
        ],
    )

    training_section["export_lora_btn"].click(
        fn=train_h.export_lora,
        inputs=[
            training_section["export_path"],
            training_section["lora_output_dir"],
        ],
        outputs=[training_section["export_status"]],
    )

    # ========== LoKr Training Tab Handlers ==========
    training_section["lokr_load_dataset_btn"].click(
        fn=train_h.load_training_dataset,
        inputs=[training_section["lokr_training_tensor_dir"]],
        outputs=[training_section["lokr_training_dataset_info"]],
    )

    training_section["start_lokr_training_btn"].click(
        fn=lokr_training_wrapper,
        inputs=[
            training_section["lokr_training_tensor_dir"],
            training_section["lokr_linear_dim"],
            training_section["lokr_linear_alpha"],
            training_section["lokr_factor"],
            training_section["lokr_decompose_both"],
            training_section["lokr_use_tucker"],
            training_section["lokr_use_scalar"],
            training_section["lokr_weight_decompose"],
            training_section["lokr_learning_rate"],
            training_section["lokr_train_epochs"],
            training_section["lokr_train_batch_size"],
            training_section["lokr_gradient_accumulation"],
            training_section["lokr_save_every_n_epochs"],
            training_section["lokr_training_shift"],
            training_section["lokr_training_seed"],
            training_section["lokr_output_dir"],
            training_section["training_state"],
        ],
        outputs=[
            training_section["lokr_training_progress"],
            training_section["lokr_training_log"],
            training_section["lokr_training_loss_plot"],
            training_section["training_state"],
        ],
    )

    training_section["stop_lokr_training_btn"].click(
        fn=train_h.stop_training,
        inputs=[training_section["training_state"]],
        outputs=[
            training_section["lokr_training_progress"],
            training_section["training_state"],
        ],
    )

    training_section["refresh_lokr_export_epochs_btn"].click(
        fn=train_h.list_lokr_export_epochs,
        inputs=[training_section["lokr_output_dir"]],
        outputs=[
            training_section["lokr_export_epoch"],
            training_section["lokr_export_status"],
        ],
    )

    training_section["export_lokr_btn"].click(
        fn=train_h.export_lokr,
        inputs=[
            training_section["lokr_export_path"],
            training_section["lokr_output_dir"],
            training_section["lokr_export_epoch"],
        ],
        outputs=[training_section["lokr_export_status"]],
    )
