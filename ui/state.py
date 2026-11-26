"""Session state helpers for ALPRO UI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import streamlit as st

from alpro.config.settings import load_pipeline_config

DEFAULT_CONFIG_PATH = "configs/pipeline.yaml"
CONFIG_STATE_KEY = "alpro_config"
CONFIG_DEFAULTS_KEY = "alpro_config_defaults"


def _load_defaults() -> Dict[str, Any]:
    cfg = load_pipeline_config(DEFAULT_CONFIG_PATH)
    # ensure expected shape
    cfg.setdefault("detector", {})
    cfg.setdefault("crop", {})
    cfg.setdefault("preprocessing", {})
    cfg.setdefault("ocr", {})
    cfg.setdefault("output", {})
    cfg.setdefault("ui", {})
    return cfg


def get_default_config() -> Dict[str, Any]:
    if CONFIG_DEFAULTS_KEY not in st.session_state:
        st.session_state[CONFIG_DEFAULTS_KEY] = _load_defaults()
    return deepcopy(st.session_state[CONFIG_DEFAULTS_KEY])


def init_config_state() -> None:
    if CONFIG_STATE_KEY not in st.session_state:
        st.session_state[CONFIG_STATE_KEY] = get_default_config()


def get_config() -> Dict[str, Any]:
    init_config_state()
    return st.session_state[CONFIG_STATE_KEY]


def set_config(cfg: Dict[str, Any]) -> None:
    st.session_state[CONFIG_STATE_KEY] = cfg


def reset_config() -> Dict[str, Any]:
    cfg = get_default_config()
    set_config(cfg)
    return cfg
