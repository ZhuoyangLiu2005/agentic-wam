"""Shared deploy pieces: wire contract, image codec, policy runtime.

Transport-free and robot-free by design — both deploy lines import from here, so the model path
and the wire format cannot drift between local (GPU on robot) and remote (GPU on cluster).
"""
