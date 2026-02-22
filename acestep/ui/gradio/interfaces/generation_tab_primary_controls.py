"""Primary generation-tab controls (mode, source, selectors, and LM code hints)."""

from typing import Any

import gradio as gr

from acestep.constants import DEFAULT_DIT_INSTRUCTION, TRACK_NAMES, VALID_LANGUAGES
from acestep.ui.gradio.help_content import create_help_button
from acestep.ui.gradio.i18n import t


def build_mode_selector_controls(initial_mode_choices: list[str]) -> dict[str, Any]:
    """Create generation mode selector and load-metadata upload controls.

    Args:
        initial_mode_choices: Mode labels available for the currently selected model family.

    Returns:
        A component map containing ``generation_mode``, ``load_file``, and ``load_file_col``.
    """

    with gr.Row(equal_height=True):
        generation_mode = gr.Radio(
            choices=initial_mode_choices,
            value="Custom",
            label=t("generation.mode_label"),
            info=t("generation.mode_info_custom"),
            elem_classes=["has-info-container"],
            scale=10,
        )
        with gr.Column(scale=1, min_width=80, elem_classes="icon-btn-wrap") as load_file_col:
            load_file = gr.UploadButton(
                t("generation.load_btn"),
                file_types=[".json"],
                file_count="single",
                variant="secondary",
                size="lg",
            )
    return {
        "generation_mode": generation_mode,
        "load_file": load_file,
        "load_file_col": load_file_col,
    }


def build_hidden_generation_state() -> dict[str, Any]:
    """Create hidden/state controls consumed by generation event wiring.

    Args:
        None.

    Returns:
        A component map containing hidden task/instruction fields and Gradio state objects.
    """

    task_type = gr.Textbox(value="text2music", visible=False, label="task_type")
    instruction_display_gen = gr.Textbox(
        label=t("generation.instruction_label"),
        value=DEFAULT_DIT_INSTRUCTION,
        interactive=False,
        lines=1,
        info=t("generation.instruction_info"),
        elem_classes=["has-info-container"],
        visible=False,
    )
    simple_sample_created = gr.State(value=False)
    lyrics_before_instrumental = gr.State(value="")
    previous_generation_mode = gr.State(value="Custom")
    return {
        "task_type": task_type,
        "instruction_display_gen": instruction_display_gen,
        "simple_sample_created": simple_sample_created,
        "lyrics_before_instrumental": lyrics_before_instrumental,
        "previous_generation_mode": previous_generation_mode,
    }


def build_simple_mode_controls() -> dict[str, Any]:
    """Create simple-mode prompt controls and sample action controls.

    Args:
        None.

    Returns:
        A component map containing simple-mode query inputs, language toggles, and action buttons.
    """

    with gr.Group(visible=False, elem_classes=["has-info-container"]) as simple_mode_group:
        create_help_button("generation_simple")
        with gr.Row(equal_height=True):
            simple_query_input = gr.Textbox(
                label=t("generation.simple_query_label"),
                placeholder=t("generation.simple_query_placeholder"),
                lines=2,
                info=t("generation.simple_query_info"),
                elem_classes=["has-info-container"],
                scale=9,
            )
            with gr.Column(scale=1):
                simple_vocal_language = gr.Dropdown(
                    choices=[
                        (lang if lang != "unknown" else "Instrumental / auto", lang)
                        for lang in VALID_LANGUAGES
                    ],
                    value="unknown",
                    allow_custom_value=True,
                    label=t("generation.simple_vocal_language_label"),
                    interactive=True,
                    scale=1,
                )
                simple_instrumental_checkbox = gr.Checkbox(
                    label=t("generation.instrumental_label"),
                    value=False,
                    scale=1,
                )
            with gr.Column(scale=1, min_width=80, elem_classes="icon-btn-wrap"):
                random_desc_btn = gr.Button(
                    t("generation.sample_btn"),
                    variant="secondary",
                    size="lg",
                )
        with gr.Row(equal_height=True):
            create_sample_btn = gr.Button(
                t("generation.create_sample_btn"),
                variant="primary",
                size="lg",
            )
    return {
        "simple_mode_group": simple_mode_group,
        "simple_query_input": simple_query_input,
        "simple_vocal_language": simple_vocal_language,
        "simple_instrumental_checkbox": simple_instrumental_checkbox,
        "random_desc_btn": random_desc_btn,
        "create_sample_btn": create_sample_btn,
    }


def build_source_track_and_code_controls() -> dict[str, Any]:
    """Create source-audio, track-selector, and LM-code hint controls.

    Args:
        None.

    Returns:
        A component map containing source audio actions, track selectors, and LM code controls.
    """

    with gr.Row(equal_height=True, visible=False) as src_audio_row:
        src_audio = gr.Audio(label=t("generation.source_audio"), type="filepath", scale=10)
        with gr.Column(scale=1, min_width=80):
            analyze_btn = gr.Button(
                t("generation.analyze_btn"),
                variant="secondary",
                size="lg",
            )

    with gr.Group(visible=False) as extract_help_group:
        create_help_button("generation_extract")
    track_name = gr.Dropdown(
        choices=TRACK_NAMES,
        value=None,
        label=t("generation.track_name_label"),
        info=t("generation.track_name_info"),
        elem_classes=["has-info-container"],
        visible=False,
    )
    with gr.Group(visible=False) as complete_help_group:
        create_help_button("generation_complete")
    complete_track_classes = gr.CheckboxGroup(
        choices=TRACK_NAMES,
        label=t("generation.track_classes_label"),
        info=t("generation.track_classes_info"),
        elem_classes=["has-info-container"],
        visible=False,
    )

    with gr.Accordion(
        t("generation.lm_codes_hints"),
        open=False,
        visible=True,
        elem_classes=["has-info-container"],
    ) as text2music_audio_codes_group:
        with gr.Row(equal_height=True):
            lm_codes_audio_upload = gr.Audio(label=t("generation.source_audio"), type="filepath", scale=3)
            text2music_audio_code_string = gr.Textbox(
                label=t("generation.lm_codes_label"),
                placeholder=t("generation.lm_codes_placeholder"),
                lines=6,
                info=t("generation.lm_codes_info"),
                elem_classes=["has-info-container"],
                scale=6,
            )
        with gr.Row():
            convert_src_to_codes_btn = gr.Button(
                t("generation.convert_codes_btn"),
                variant="secondary",
                size="sm",
                scale=1,
            )
            transcribe_btn = gr.Button(
                t("generation.transcribe_btn"),
                variant="secondary",
                size="sm",
                scale=1,
            )

    return {
        "src_audio_row": src_audio_row,
        "src_audio": src_audio,
        "analyze_btn": analyze_btn,
        "extract_help_group": extract_help_group,
        "track_name": track_name,
        "complete_help_group": complete_help_group,
        "complete_track_classes": complete_track_classes,
        "text2music_audio_codes_group": text2music_audio_codes_group,
        "lm_codes_audio_upload": lm_codes_audio_upload,
        "text2music_audio_code_string": text2music_audio_code_string,
        "convert_src_to_codes_btn": convert_src_to_codes_btn,
        "transcribe_btn": transcribe_btn,
    }
