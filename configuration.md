# Configuration Guide

This document provides detailed information about configuring DeliveryBench experiments.

## Configuration Files

Experimental settings are split into two configuration files:

### 1. `vlm_delivery/input/experiment_config.json`

**Frequently modified experimental parameters** - you should edit this file for most experiments.

#### `gym_env` - Environment settings

- **`base_dir`** (string or null): Base directory path where you downloaded/placed the DeliveryBench repository (e.g., `~/benchmark/DeliveryBench`).

  
- **`ue_ip`** (string): Unreal Engine server IP address. Default: `"127.0.0.1"` (localhost). Change this if running the UE server on a remote machine.

- **`ue_port`** (integer): Unreal Engine server port number. Default: `9000`. Must match the port configured in the UE server's `gym_citynav/Binaries/Linux/unrealcv.ini` file.

- **`map_name`** (string): Name of the map to use. Must exactly match the map name used when launching `gym_citynav.exe`. Examples: `"medium-city-22"`, `"small-city-11"`, `"large-city-26"`. See `maps/` for the full list of available cities.

#### `lifecycle` - Episode lifecycle settings

- **`export_path`** (string): Directory path for exporting episode-level results. After an episode completes, a JSON file containing detailed statistics (earnings, deliveries completed, time taken, etc.) will be saved to this directory. Example: `"outputs/final_results"`.

#### Other settings

- **`trajectory_output_dir`** (string): Directory for logging detailed agent trajectories. Each step includes the agent's observations, actions, and state. Useful for detailed analysis and debugging. Example: `"outputs/trajectories"`.

- **`agent_count`** (integer): Number of delivery agents to spawn in the environment. Only used when `multi_agent` is enabled. Default: `1`.

- **`multi_agent`** (boolean): Enable/disable multi-agent mode. When `true`, multiple agents can operate simultaneously in the same environment. Requires `agent_count > 1`.

---

### 2. `vlm_delivery/input/game_mechanics_config.json`

**Game mechanics parameters** - not recommended to modify unless you understand the implications. These parameters define the core game mechanics and should match the experimental setup in our paper.

#### Movement and Transportation

- **`avg_speed_cm_s`** (object): Average movement speed in centimeters per second for different transportation modes:
  - `walk` (integer): Walking speed (default: 200 cm/s = 2 m/s)
  - `e-scooter` (integer): Electric scooter speed (default: 600 cm/s = 6 m/s)
  - `drag_scooter` (integer): Dragging scooter speed when battery is dead (default: 150 cm/s = 1.5 m/s)
  - `car` (integer): Car speed (default: 1200 cm/s = 12 m/s)
  - `bus` (integer): Bus speed (default: 1000 cm/s = 10 m/s)

- **`pace_scales`** (object): Speed multipliers for different movement paces. The agent can choose accelerated/normal/decelerated pace. The selected pace scales movement speed, and also scales per-meter consumption proportionally (e.g., stamina, and e-scooter battery when riding).
  - `accel` (float): Accelerated pace (default: 1.25)
  - `normal` (float): Normal pace (default: 1.0)
  - `decel` (float): Decelerated pace (default: 0.75)

- **`bus`** (object): Bus system configuration:
  - `waiting_time_s` (integer): Time in seconds to wait at each bus stop (default: 360s = 6 minutes)
  - `speed_cm_s` (integer): Bus movement speed in cm/s (default: 1000 cm/s = 10 m/s)
  - `num_buses` (integer): Number of buses in the system (default: 1)

- **`escooter_defaults`** (object): Default e-scooter settings:
  - `initial_battery_pct` (integer): Starting battery percentage when obtaining a scooter (default: 50%)
  - `charge_rate_pct_per_min` (float): Battery charging rate in percentage per minute (default: 7.5% per minute)
  - `charge_target_pct` (integer): Default target battery percentage when using CHARGE_ESCOOTER action without explicitly specifying `target_pct` (default: 100%). The agent will charge the e-scooter until reaching this battery level. Can be overridden by specifying `target_pct` in the CHARGE_ESCOOTER action (e.g., `CHARGE_ESCOOTER(target_pct=80)`).

- **`rent_car_defaults`** (object): Default car rental settings:
  - `avg_speed_cm_s` (integer): Car speed in cm/s (default: 1200 cm/s = 12 m/s)
  - `rate_per_min` (float): Rental cost per minute in dollars (default: 1.0)

#### Energy and Health

- **`energy_pct_decay_per_m_by_mode`** (object): Energy percentage lost per meter traveled, by transportation mode:
  - `walk` (float): Energy decay when walking (default: 0.08% per meter)
  - `drag_scooter` (float): Energy decay when dragging scooter (default: 0.1% per meter)
  - `e-scooter` (float): Energy decay when riding e-scooter (default: 0.01% per meter)
  - `car` (float): Energy decay when driving car (default: 0.008% per meter)
  - `bus` (float): Energy decay when riding bus (default: 0.006% per meter)

- **`scooter_batt_decay_pct_per_m`** (float): E-scooter battery percentage lost per meter traveled (default: 0.04% per meter).

- **`energy_pct_max`** (integer): Maximum energy percentage (default: 100).

- **`rest_rate_pct_per_min`** (float): Energy recovery rate when resting, in percentage per minute (default: 7.5% per minute).

- **`rest_target_pct`** (integer): Default target energy percentage when using REST action without explicitly specifying `target_pct` (default: 100%). The agent will rest until reaching this energy level. Can be overridden by specifying `target_pct` in the REST action (e.g., `REST(target_pct=80)`).

- **`low_energy_threshold_pct`** (integer): Energy threshold below which the agent is considered low on energy (default: 30%). The agent may need to rest or use energy drinks when below this threshold.

- **`hospital_duration_s`** (integer): Time in seconds the agent must spend in the hospital when rescued (default: 900s = 15 minutes). See Economic Parameters section for the rescue fee cost.

#### Food and Orders

- **`ambient_temp_c`** (float): Ambient temperature in Celsius (default: 22.0°C). Food items cool down towards this temperature over time. Affects food quality and delivery requirements.

- **`k_food_per_s`** (float): Food temperature decay rate per second (default: 0.000555... ≈ 1/1800, meaning food cools down over ~30 minutes). Higher values mean faster cooling. Food temperature relaxes exponentially towards `ambient_temp_c` using this decay constant.

#### Items and Purchases

- **`items`** (object): Purchasable items and their effects:
  - `energy_drink` (object): Energy drink item
    - `energy_gain_pct` (integer): Energy percentage gained when consumed (default: 50%)
  - `escooter_battery_pack` (object): E-scooter battery pack item
    - `target_charge_pct` (integer): Battery percentage after using the pack (default: 100%)

#### Economic Parameters

- **`initial_earnings`** (float): Starting money balance in dollars (default: 100.0).

- **`charge_price_per_percent`** (float): Cost in dollars to charge the e-scooter battery by 1% (default: 0.05).

- **`hospital_rescue_fee`** (float): Cost in dollars for hospital rescue service (default: 5.0). See Energy and Health section for hospital duration.

- **`settlement`** (object): Settlement/payment parameters (currently empty, reserved for future use).

#### Spatial and Routing

- **`tolerance_cm`** (object): Distance tolerances in centimeters for different operations:
  - `nearby` (float): Distance threshold for "nearby" checks (default: 500.0 cm = 5 m)
  - `arrive` (float): Distance threshold for arrival detection (default: 500.0 cm = 5 m)
  - `scooter` (float): Distance threshold for scooter operations (default: 500.0 cm = 5 m)
  - `tempbox` (float): Distance threshold for temporary storage box operations (default: 500.0 cm = 5 m)

- **`routing`** (object): Routing system parameters:
  - `snap_cm` (float): Distance in centimeters to snap waypoints to roads (default: 120.0 cm = 1.2 m)

- **`map`** (object): Map generation and layout parameters:
  - `sidewalk_offset_cm` (integer): Sidewalk offset from road center in cm (default: 1700 cm = 17 m)
  - `drive_lane_offset_cm` (integer): Driving lane offset in cm (default: 1000 cm = 10 m)
  - `drive_waypoint_spacing_cm` (integer): Spacing between driving waypoints in cm (default: 5000 cm = 50 m)

#### UI and Visualization

- **`ui`** (object): User interface settings:
  - `show_path_ms` (integer): Duration in milliseconds to display agent paths in the viewer (default: 2000 ms = 2 seconds)

- **`map_snapshot_limits`** (object): Limits for map snapshot generation (used in agent observations):
  - `next` (integer): Number of next waypoints to show (default: 20)
  - `s` (integer): Number of seconds of future path to show (default: 40)
  - `poi` (integer): Maximum distance in meters for POI (Points of Interest) visibility (default: 800 m)

#### VLM (Vision-Language Model) Settings

- **`vlm`** (object): VLM inference settings:
  - `next_action_delay_ms` (integer): Delay in milliseconds before requesting the next action from the VLM (default: 300 ms)
  - `retry_max` (integer): Maximum number of retries for VLM API calls on failure (default: 5)

#### Simulation Timing

- **`gym_env`** (object): Gym environment timing parameters:
  - `sim_tick_ms` (integer): Simulation tick interval in milliseconds (default: 100 ms, i.e., 10 ticks per second)
  - `vlm_pump_ms` (integer): Interval in milliseconds between VLM decision requests (default: 100 ms)

#### Lifecycle Limits

- **`lifecycle`** (object): Episode lifecycle termination conditions:
  - `duration_hours` (integer): Maximum episode duration in simulated hours (default: 2 hours). Episode ends when this time is reached.
  - `realtime_stop_hours` (integer): Maximum real-world (wall-clock) time in hours (default: 2 hours). Episode ends if this real-time limit is exceeded.
  - `vlm_call_limit` (integer): Maximum number of VLM API calls per episode (default: 200). Episode ends when this limit is reached.

#### Unreal Engine Models

- **`ue_models`** (object): Paths to Unreal Engine blueprint models (internal use, do not modify):
  - `delivery_man` (string): Path to delivery man character model
  - `delivery_manager` (string): Path to delivery manager system
  - `scooter` (string): Path to scooter vehicle model
  - `customer` (string): Path to customer character model