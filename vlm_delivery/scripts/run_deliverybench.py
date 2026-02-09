# scripts/run_deliverybench.py
# -*- coding: utf-8 -*-

import json
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import threading
import traceback
from pathlib import Path

# Auto-detect the Food-Delivery-Bench repo root.
cwd = Path.cwd().resolve()
base_dir = None
for p in [cwd, *cwd.parents]:
    if (p / "vlm_delivery").is_dir() and (p / "simworld").is_dir():
        base_dir = p
        break
    candidate = p / "Food-Delivery-Bench"
    if (candidate / "vlm_delivery").is_dir() and (candidate / "simworld").is_dir():
        base_dir = candidate
        break

if base_dir is None:
    raise RuntimeError("Cannot auto-detect Food-Delivery-Bench root.")

base_dir = str(base_dir)
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "simworld"))

from vlm_delivery.gym_like_interface.gym_like_interface import DeliveryBenchGymEnv
from vlm_delivery.vlm.base_model import BaseModel
from vlm_delivery.utils.trajectory_recorder import save_text
from vlm_delivery.utils.vlm_runtime import (
    sanitize_filename,
    vlm_decide_action_with_retry,
    vlm_collect_images,
    export_vlm_images_debug_once,
)


def build_vlm_client(base_dir: str) -> BaseModel:
    """Load VLM client configuration from `vlm_delivery/input/models.json`."""
    models_path = Path(base_dir) / "vlm_delivery" / "input" / "models.json"
    with models_path.open("r", encoding="utf-8") as f:
        models_cfg = json.load(f) or {}

    agents = models_cfg.get("agents", {}) or {}
    default = models_cfg.get("default", {}) or {}
    agent_cfg = agents.get("1", {}) or {}
    cfg = dict(default)
    cfg.update(agent_cfg)

    provider = (cfg.get("provider") or "openai").lower()
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or ""
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY") or ""
    api_key = openai_key if provider == "openai" else openrouter_key
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider={provider}. "
            "Set OPENAI_API_KEY or OPENROUTER_API_KEY (or *_KEY)."
        )

    return BaseModel(
        url=cfg.get("url", "https://api.openai.com/v1"),
        api_key=api_key,
        model=cfg.get("model", "gpt-4o-mini"),
    )


def make_vlm_caller(dm, vlm: BaseModel, env=None):
    """Factory: return a `call_vlm(prompt) -> str` closure.

    When env is provided, image collection runs on the Qt main thread via
    env._invoker so we never touch Qt from a worker thread (avoids crash).
    """

    def _call(prompt: str) -> str:
        if env is not None and getattr(env, "_invoker", None) is not None:
            box = env._invoker.call(lambda: vlm_collect_images(dm))
            images = box["result"] if box.get("ok") else [None, None, None]
        else:
            images = vlm_collect_images(dm)
        return str(vlm.generate(user_prompt=prompt, images=images))

    return _call


def main(max_steps: int = 20):
    exp_cfg_path = os.path.join(base_dir, "vlm_delivery", "input", "experiment_config.json")
    with open(exp_cfg_path, "r", encoding="utf-8") as f:
        exp_cfg = json.load(f) or {}
    gym_env_cfg = exp_cfg.get("gym_env", {}) or {}

    env = DeliveryBenchGymEnv(
        base_dir=base_dir,
        ue_ip=gym_env_cfg.get("ue_ip", "127.0.0.1"),
        ue_port=int(gym_env_cfg.get("ue_port", 9000)),
        sim_tick_ms=100,
        vlm_pump_ms=100,
        enable_viewer=True,
        map_name=gym_env_cfg.get("map_name", "medium-city-22"),
        max_steps=max_steps,
    )

    env.bootstrap_qt()

    # Run reset() on the main thread so all Qt (viewer, timers) is created here.
    obs, info = env.reset(seed=0)
    print("reset info:", info)
    print("obs:", obs)

    if not env.dms:
        raise RuntimeError("No DeliveryMan instances found after reset().")
    dm = env.dms[0]

    def rl_loop():
        try:
            vlm = build_vlm_client(base_dir)
            dm._vlm_client = vlm  # for export_vlm_images_debug_once (model name in filenames)
            call_vlm = make_vlm_caller(dm, vlm, env)

            for step_i in range(1, max_steps + 1):
                # Save debug images + prompt for this step (must run on Qt main thread)
                env._invoker.call(lambda: export_vlm_images_debug_once(dm))

                dm_action, raw_text, last_err = vlm_decide_action_with_retry(
                    dm,
                    call_vlm,
                    max_retries=2,
                )
                print(f"\n=== Step {step_i} ===")
                print("Chosen action:", dm_action)
                if last_err:
                    print("VLM error hint:", last_err)

                try:
                    model_safe = sanitize_filename(getattr(vlm, "model", "unknown_model"))
                    step_idx = step_i - 1  # 0-based, same as prompt/images
                    filename = f"{model_safe}_{step_idx}_output.txt"
                    save_text(dm.run_dir, filename, raw_text, encoding="utf-8")
                except Exception:
                    pass

                obs, r, term, trunc, info2 = env.step(dm_action)
                print("info:", info2)
                print("reward:", r, "done:", term, "truncated:", trunc)

                if info2.get("error"):
                    print("ENV ERROR:", info2["error"])
                    if info2.get("parse_exc"):
                        print("PARSE TRACEBACK:", info2["parse_exc"])
                    break

                if term or trunc:
                    break

        except Exception as e:
            print("[RL] Exception:", e)
            traceback.print_exc()

        finally:
            try:
                env.close()
            except Exception:
                pass
            try:
                if getattr(env, "_app", None) is not None:
                    env._app.quit()
            except Exception:
                pass

    threading.Thread(target=rl_loop, daemon=True).start()
    env.run_qt_loop()


if __name__ == "__main__":
    main()
