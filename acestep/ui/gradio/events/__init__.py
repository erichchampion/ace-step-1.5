"""
Gradio UI Event Handlers Module
Main entry point for setting up all event handlers
"""
# Import handler modules
from . import generation_handlers as gen_h
from . import results_handlers as res_h
from .wiring import (
    GenerationWiringContext,
    TrainingWiringContext,
    build_mode_ui_outputs,
    register_generation_batch_navigation_handlers,
    register_generation_metadata_handlers,
    register_generation_mode_handlers,
    register_generation_run_handlers,
    register_results_aux_handlers,
    register_generation_service_handlers,
    register_training_dataset_builder_handlers,
    register_training_dataset_load_handler,
    register_training_preprocess_handler,
    register_training_run_handlers,
)


def setup_event_handlers(demo, dit_handler, llm_handler, dataset_handler, dataset_section, generation_section, results_section):
    """Setup generation/results event wiring for the Gradio UI.

    Args:
        demo (Any): Root Gradio demo/container used to register events.
        dit_handler (Any): Inference service used by generation/results callbacks.
        llm_handler (Any): LLM service used by metadata/text callbacks.
        dataset_handler (Any): Dataset service used by generation wiring.
        dataset_section (dict[str, Any]): Dataset UI component map.
        generation_section (dict[str, Any]): Generation UI component map.
        results_section (dict[str, Any]): Results UI component map.

    Local wiring values:
        wiring_context (GenerationWiringContext): Shared typed context for
            generation/results wiring helper modules.
        auto_checkbox_inputs (list[Any]): Ordered metadata fields used for
            auto-checkbox synchronization; forwarded to
            register_generation_metadata_handlers and
            register_generation_mode_handlers.
        auto_checkbox_outputs (list[Any]): Ordered auto toggles plus derived
            metadata outputs returned by register_generation_service_handlers;
            forwarded to register_generation_metadata_handlers and
            register_generation_mode_handlers.
        mode_ui_outputs (list[Any]): Ordered mode-UI outputs from
            build_mode_ui_outputs; forwarded to
            register_generation_mode_handlers and register_results_aux_handlers.

    Returns:
        None: Registers event handlers in-place on the supplied components.
    """
    wiring_context = GenerationWiringContext(
        demo=demo,
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        dataset_handler=dataset_handler,
        dataset_section=dataset_section,
        generation_section=generation_section,
        results_section=results_section,
    )
    
    auto_checkbox_inputs, auto_checkbox_outputs = register_generation_service_handlers(
        wiring_context
    )
    mode_ui_outputs = build_mode_ui_outputs(wiring_context)
    register_generation_metadata_handlers(
        wiring_context,
        auto_checkbox_inputs=auto_checkbox_inputs,
        auto_checkbox_outputs=auto_checkbox_outputs,
    )

    register_generation_mode_handlers(
        wiring_context,
        mode_ui_outputs=mode_ui_outputs,
        auto_checkbox_inputs=auto_checkbox_inputs,
        auto_checkbox_outputs=auto_checkbox_outputs,
    )

    # ========== Load/Save Metadata ==========
    generation_section["load_file"].upload(
        fn=lambda file_obj: gen_h.load_metadata(file_obj, llm_handler),
        inputs=[generation_section["load_file"]],
        outputs=[
            generation_section["task_type"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["vocal_language"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["inference_steps"],
            generation_section["guidance_scale"],
            generation_section["seed"],
            generation_section["random_seed_checkbox"],
            generation_section["use_adg"],
            generation_section["cfg_interval_start"],
            generation_section["cfg_interval_end"],
            generation_section["shift"],
            generation_section["infer_method"],
            generation_section["custom_timesteps"],
            generation_section["audio_format"],
            generation_section["lm_temperature"],
            generation_section["lm_cfg_scale"],
            generation_section["lm_top_k"],
            generation_section["lm_top_p"],
            generation_section["lm_negative_prompt"],
            generation_section["use_cot_metas"],  # Added: use_cot_metas
            generation_section["use_cot_caption"],
            generation_section["use_cot_language"],
            generation_section["audio_cover_strength"],
            generation_section["cover_noise_strength"],
            generation_section["think_checkbox"],
            generation_section["text2music_audio_code_string"],
            generation_section["repainting_start"],
            generation_section["repainting_end"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["instrumental_checkbox"],  # Added: instrumental_checkbox
            results_section["is_format_caption_state"]
        ]
    ).then(
        fn=gen_h.uncheck_auto_for_populated_fields,
        inputs=auto_checkbox_inputs,
        outputs=auto_checkbox_outputs,
    )
    
    # Save buttons for all 8 audio outputs
    download_existing_js = """(current_audio, batch_files) => {
    // Debug: print what the input actually is
    console.log("👉 [Debug] Current Audio Input:", current_audio);
    
    // 1. Safety check
    if (!current_audio) {
        console.warn("⚠️ No audio selected or audio is empty.");
        return;
    }
    if (!batch_files || !Array.isArray(batch_files)) {
        console.warn("⚠️ Batch file list is empty/not ready.");
        return;
    }

    // 2. Smartly extract path string
    let pathString = "";
    
    if (typeof current_audio === "string") {
        // Case A: direct path string received
        pathString = current_audio;
    } else if (typeof current_audio === "object") {
        // Case B: an object is received, try common properties
        // Gradio file objects usually have path, url, or name
        pathString = current_audio.path || current_audio.name || current_audio.url || "";
    }

    if (!pathString) {
        console.error("❌ Error: Could not extract a valid path string from input.", current_audio);
        return;
    }

    // 3. Extract Key (UUID)
    // Path could be /tmp/.../uuid.mp3 or url like /file=.../uuid.mp3
    let filename = pathString.split(/[\\\\/]/).pop(); // get the filename
    let key = filename.split('.')[0]; // get UUID without extension

    console.log(`🔑 Key extracted: ${key}`);

    // 4. Find matching file(s) in the list
    let targets = batch_files.filter(f => {
        // Also extract names from batch_files objects
        // f usually contains name (backend path) and orig_name (download name)
        const fPath = f.name || f.path || ""; 
        return fPath.includes(key);
    });

    if (targets.length === 0) {
        console.warn("❌ No matching files found in batch list for key:", key);
        alert("Batch list does not contain this file yet. Please wait for generation to finish.");
        return;
    }

    // 5. Trigger download(s)
    console.log(`🎯 Found ${targets.length} files to download.`);
    targets.forEach((f, index) => {
        setTimeout(() => {
            const a = document.createElement('a');
            // Prefer url (frontend-accessible link), otherwise try data
            a.href = f.url || f.data; 
            a.download = f.orig_name || "download";
            a.style.display = 'none';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }, index * 1000); // 300ms interval to avoid browser blocking
    });
}
"""
    for btn_idx in range(1, 9):
        results_section[f"save_btn_{btn_idx}"].click(
            fn=None,
            inputs=[
                results_section[f"generated_audio_{btn_idx}"],
                results_section["generated_audio_batch"],
            ],
        js=download_existing_js  # Run the above JS
    )
    register_results_aux_handlers(
        wiring_context,
        mode_ui_outputs=mode_ui_outputs,
    )
    register_generation_run_handlers(wiring_context)
    register_generation_batch_navigation_handlers(wiring_context)
    
    # ========== Restore Parameters Handler ==========
    results_section["restore_params_btn"].click(
        fn=res_h.restore_batch_parameters,
        inputs=[
            results_section["current_batch_index"],
            results_section["batch_queue"]
        ],
        outputs=[
            generation_section["text2music_audio_code_string"],
            generation_section["captions"],
            generation_section["lyrics"],
            generation_section["bpm"],
            generation_section["key_scale"],
            generation_section["time_signature"],
            generation_section["vocal_language"],
            generation_section["audio_duration"],
            generation_section["batch_size_input"],
            generation_section["inference_steps"],
            generation_section["lm_temperature"],
            generation_section["lm_cfg_scale"],
            generation_section["lm_top_k"],
            generation_section["lm_top_p"],
            generation_section["think_checkbox"],
            generation_section["use_cot_caption"],
            generation_section["use_cot_language"],
            generation_section["allow_lm_batch"],
            generation_section["track_name"],
            generation_section["complete_track_classes"],
            generation_section["enable_normalization"],
            generation_section["normalization_db"],
            generation_section["latent_shift"],
            generation_section["latent_rescale"],
        ]
    )
    
    # ========== LRC Display Change Handlers ==========
    # NEW APPROACH: Use lrc_display.change() to update audio subtitles
    # This decouples audio value updates from subtitle updates, avoiding flickering.
    #
    # When lrc_display text changes (from generate, LRC button, or manual edit):
    # 1. lrc_display.change() is triggered
    # 2. update_audio_subtitles_from_lrc() parses LRC and updates audio subtitles
    # 3. Audio value is NEVER updated here - only subtitles
    for lrc_idx in range(1, 9):
        results_section[f"lrc_display_{lrc_idx}"].change(
            fn=res_h.update_audio_subtitles_from_lrc,
            inputs=[
                results_section[f"lrc_display_{lrc_idx}"],
                # audio_duration not needed - parse_lrc_to_subtitles calculates end time from timestamps
            ],
            outputs=[
                results_section[f"generated_audio_{lrc_idx}"],  # Only updates subtitles, not value
            ]
        )


def setup_training_event_handlers(demo, dit_handler, llm_handler, training_section):
    """Setup event handlers for the training tab (dataset builder and LoRA training)"""
    training_context = TrainingWiringContext(
        demo=demo,
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        training_section=training_section,
    )
    
    # ========== Load Existing Dataset (Top Section) ==========

    # Load existing dataset JSON at the top of Dataset Builder
    register_training_dataset_load_handler(
        training_context,
        button_key="load_json_btn",
        path_key="load_json_path",
        status_key="load_json_status",
    )
    # ========== Dataset Builder Handlers ==========
    register_training_dataset_builder_handlers(training_context)

    # ========== Preprocess Handlers ==========
    
    # Load existing dataset JSON for preprocessing
    # This also updates the preview section so users can view/edit samples
    register_training_dataset_load_handler(
        training_context,
        button_key="load_existing_dataset_btn",
        path_key="load_existing_dataset_path",
        status_key="load_existing_status",
    )
    
    # Preprocess dataset to tensor files
    register_training_preprocess_handler(training_context)
    register_training_run_handlers(training_context)
