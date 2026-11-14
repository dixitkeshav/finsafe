"""Frozen, provenance-bound SafeTune inference interventions."""
import json
from pathlib import Path
from .resume import sha256_file


def adapter_sha(path):
    return sha256_file(Path(path)/'adapter_model.safetensors') if path else None


def load_steering(model, folder, model_id, revision, adapter, alpha):
    from safetensors.torch import load_file
    from safetune.interventions.steer.refusal_direction import RefusalDirectionModel
    folder=Path(folder)
    cfg=json.loads((folder/'config.json').read_text())
    expected={'model':model_id,'revision':revision,'adapter_sha256':adapter_sha(adapter)}
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Steering artifact does not match the model/adapter')
    if cfg['direction_sha256']!=sha256_file(folder/'direction.safetensors'):
        raise ValueError('Steering tensor checksum mismatch')
    return RefusalDirectionModel(model,load_file(folder/'direction.safetensors')['direction'],
        mode='steer',strength=alpha,layers=[cfg['layer']])
