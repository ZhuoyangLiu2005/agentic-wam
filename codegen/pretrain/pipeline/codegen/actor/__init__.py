"""VLM actor: router client, frame media, the coder prompt builder, repair loop.

NO eager submodule imports here: ``prompts`` resolves the ACTIVE embodiment at
import time (``codegen.api.active``), and the CLI entry points must be able to
import the vocabulary-free ``client`` module BEFORE setting
``CODEGEN_EMBODIMENT``. Import submodules explicitly.
"""
