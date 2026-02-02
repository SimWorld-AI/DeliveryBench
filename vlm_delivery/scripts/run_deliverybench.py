# run_gym_qt_route_a.py
import json
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import threading
import traceback
from pathlib import Path

# Auto-detect the Food-Delivery-Bench repo root.
cwd = Path.cwd().resolve()
base_dir = None
for p in [cwd, *cwd.parents]:
    # 1) Current directory is the repo root.
    if (p / "vlm_delivery").is_dir() and (p / "simworld").is_dir():
        base_dir = p
        break
    # 2) Repo root is a direct child of the current path.
    candidate = p / "Food-Delivery-Bench"
    if (candidate / "vlm_delivery").is_dir() and (candidate / "simworld").is_dir():
        base_dir = candidate
        break

if base_dir is None:
    raise RuntimeError("Cannot auto-detect Food-Delivery-Bench root.")

base_dir = str(base_dir)
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "simworld"))

from PyQt5.QtCore import QTimer
from vlm_delivery.gym_like_interface.gym_like_interface import DeliveryBenchGymEnvQtRouteA


def main():
    exp_cfg_path = os.path.join(base_dir, "vlm_delivery", "input", "experiment_config.json")
    with open(exp_cfg_path, "r", encoding="utf-8") as f:
        exp_cfg = json.load(f) or {}
    gym_env_cfg = exp_cfg.get("gym_env", {}) or {}

    env = DeliveryBenchGymEnvQtRouteA(
        base_dir=base_dir,
        ue_ip=gym_env_cfg.get("ue_ip", "127.0.0.1"),
        ue_port=int(gym_env_cfg.get("ue_port", 9099)),
        sim_tick_ms=100,
        vlm_pump_ms=100,
        enable_viewer=True,  # If unstable, try False to isolate viewer issues.
        map_name=gym_env_cfg.get("map_name", "medium-city-22"),
        max_steps=20,
    )

    # 1) Must run on the main thread: create QApplication + invoker.
    env.bootstrap_qt()

    def rl_loop():
        try:
            # 2) Run reset/step on a worker thread.
            obs, info = env.reset(seed=0)
            print("reset info:", info)
            print("obs:", obs)

            for step_i in range(1, 999999):
                obs, r, term, trunc, info2 = env.step(None)
                print(f"[RL] step={step_i} info:", info2)

                # Print any error with tracebacks.
                if info2.get("error"):
                    print("STEP ERROR:", info2["error"])
                    if info2.get("dispatch_exc"):
                        print("DISPATCH TRACEBACK:\n", info2["dispatch_exc"])
                    if info2.get("enqueue_exc"):
                        print("ENQUEUE TRACEBACK:\n", info2["enqueue_exc"])
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

    # 3) Start the RL thread after the Qt loop is ready.
    QTimer.singleShot(0, lambda: threading.Thread(target=rl_loop, daemon=True).start())

    # 4) Run the Qt event loop on the main thread.
    env.run_qt_loop()


if __name__ == "__main__":
    main()