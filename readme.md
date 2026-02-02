# DeliveryBench: Can Agents Earn Profit in Real World?

</p> <p align="center"> 📄 <a href="https://arxiv.org/abs/2512.19234">Paper</a> &nbsp; | &nbsp; 🏠 <a href="https://deliverybench.github.io/">Website</a> &nbsp; | &nbsp; 🏙️ <a href="#">City Environments (Download)</a> &nbsp  </p> <p align="center"> 

<p align="center">
  <a href="https://mao1207.github.io/">Lingjun Mao<sup>1</sup></a>, 
  <a href="https://renjw02.github.io/">Jiawei Ren<sup>1</sup></a>, 
  <a href="https://lancelot39.github.io/">Kun Zhou<sup>1</sup></a>, 
  <a href="https://chenjix.github.io/">Jixuan Chen<sup>1</sup></a>, 
  <a href="https://mars-tin.github.io/">Ziqiao Ma<sup>2</sup></a>, 
  <a href="https://lianhui.ucsd.edu/">Lianhui Qin<sup>1</sup></a>
</p>

<p align="center">
  <sup>1</sup> University of California, San Diego &nbsp; &nbsp; 
  <sup>2</sup> University of Michigan
</p>

## 🎬 Demonstration
<p align="center">
  <a href="https://youtu.be/XizDL9Hv6q0?si=1fQivfG0ExmQQJEl" target="_blank" rel="noopener noreferrer">
    <img
      src="https://img.youtube.com/vi/XizDL9Hv6q0/0.jpg"
      alt="SimWorld Demo Video"
      width="600"
    />
  </a>
</p>

<p align="center">
  <a href="https://youtu.be/XizDL9Hv6q0?si=1fQivfG0ExmQQJEl" target="_blank" rel="noopener noreferrer">
    ▶ Watch the demo on YouTube
  </a>
</p>

## 💡 Introduction

<p align="center">
  <img src="https://www.dropbox.com/scl/fi/5o17v7mct4ew2c6n2f4be/deliverybench.png?rlkey=3dhazn7ewf1fna6krhg00h4ks&st=eb2whx62&raw=1" alt="image" />
</p>

**DeliveryBench** is a city-scale embodied benchmark that evaluates whether VLM agents can earn profit under realistic, long-horizon constraints. Agents act as autonomous couriers in 3D cities, accepting and completing delivery orders across multiple in-game hours. They must manage resources (e.g., stamina, e-scooter battery), adapt to changing conditions, and balance efficiency, timing, and cost. When multiple agents coexist, they also face social dynamics such as competition and collaboration. By jointly modeling economic, physical, and social dynamics within a unified embodied environment, DeliveryBench provides a realistic, action-driven setting to test whether VLM-based agents can plan and act strategically to improve financial outcomes.

## 🔍 Key Features

<p align="center">
  <img src="https://www.dropbox.com/scl/fi/bku8d5lw9qh90mi8mq7bp/comparison.png?rlkey=j99bpymop97gatmsbiw861s2c&st=6svyhdk2&raw=1" alt="image" />
</p>


Compared with prior embodied benchmarks, DeliveryBench supports long-horizon tasks (several in-game hours; typically > 100 action steps) with multi-dimensional real-world constraints, covering:

**⏳ Time Constraints**: Tasks have deadlines and time windows that determine when they can be performed.
Agents must schedule actions to avoid late deliveries and make efficient use of limited working time.

**🗺️ Spatial Constraints**: Some actions are only valid at specific locations, so agents must navigate 3D cities and visit the right POIs in the right order (e.g., restaurants, charging stations).

**🔋 Resource Constraints**: Agents must manage consumables such as stamina, vehicle battery, and cash to stay operational, sometimes transforming one resource into another (e.g., buying an energy drink to restore stamina).

**⚙️ Physical Constraints**: Environmental dynamics (e.g., temperature, motion, collisions) affect food quality, requiring agents to consider item fragility and perishability in route planning.

**💵 Economic Constraints**: Agents earn income but also pay operational costs (e.g., recharging, renting, buying supplies), forcing them to balance short-term expenses against long-term profit.

**🤝 Social Constraints**: In multi-agent settings, couriers collaborate and compete for limited opportunities (e.g., high-value orders, charging spots), shaping both strategy and outcomes.

## 🚀 Setup
### Project Structure
```bash
evaluation/             # Evaluation and analysis utilities

maps/                   # Test city maps and map configs used in benchmark tasks

simworld/               # Core simulation backend (Python API for the UE-based SimWorld engine);
                        # see the SimWorld repo for detailed documentation

vlm_delivery/           # VLM-based delivery agent implementation and runtime
    actions/            # Concrete agent actions (e.g., ACCEPT_ORDER, MOVE_TO, BUY)
    base/               # Shared base classes (e.g., timers, type definitions)
    communicator/       # Interface between Python and the Unreal Engine environment (UnrealCV API)
    entities/           # Entity classes (e.g., DeliveryMan, Order, vehicles)
    gameplay/           # Runtime logic such as run_recorders, prompt construction
    gym_like_interface/ # Gym-style wrappers and RL-compatible environment interface
    input/              # Task and environment configuration (food types, agent count, etc.)
    map/                # Map abstractions, waypoint systems, and visualization utilities used by the agent
    scripts/            # Test scripts to quickly run DeliveryBench
    utils/              # Helper utilities and common functions
    vlm/                # Core VLM wrapper classes and model interfaces


.gitignore
README.md
```

### Installation

#### Step 1. Set up the Python client
Make sure to use Python 3.10 or later.

```bash
git clone https://github.com/mao1207/DeliveryBench.git
cd DeliveryBench
conda create -n deliverybench python=3.10
conda activate deliverybench
pip install -e .
```

#### Step 2. Download the DeliveryBench Unreal Engine (UE) server

Our DeliveryBench UE server is built on top of [SimWorld](https://github.com/SimWorld-AI/SimWorld). Download the DeliveryBench UE server executable from HuggingFace and unzip it. This server renders the 3D city environment and runs the underlying simulation for the delivery tasks. Choose the package that matches your operating system.

- **Windows:** [DeliveryBench Windows](https://huggingface.co/datasets/SimWorld-AI/DeliveryBench/blob/main/Windows.zip)

- **Linux:** [DeliveryBench Linux](https://huggingface.co/datasets/SimWorld-AI/DeliveryBench/blob/main/Linux.zip)

### Quick Start

#### Step 1. Launch the DeliveryBench UE server

Start the DeliveryBench UE server first, then run the Python examples. From the extracted UE server package directory:

- **Windows:** double-click `gym_citynav.exe`, or launch it from the command line:

  ```bash
  gym_citynav.exe <map_name>
  ```

- **Linux:** run:

  ```bash
  ./gym_citynav.sh <map_name> -RenderOffscreen
  ```

Supported `map_name` options include (examples):

```
small-city-11, medium-city-22, large-city-26
```

See `maps/` for the full list of available cities.


#### Step 2. Configure experiments

Configuration files are under `vlm_delivery/input/`:

* `experiment_config.json`: Experiment-facing settings (e.g., which map to run). Edit this file for most runs.
* `game_mechanics_config.json`: Game mechanics parameters such as vehicle speed/cost and e-scooter charging rate. We recommend keeping this file unchanged to stay aligned with our default experimental setup.

In `experiment_config.json`, make sure the following fields are set correctly:

* `map_name`: Must match the map name used when launching `gym_citynav.exe`
* `ue_port`: Must match the UE server port (default: 9000)
* `multi_agent`: Enable/disable multi-agent mode
* `agent_count`: Number of courier agents to spawn (only used when `multi_agent` is enabled)


For detailed configuration documentation, see [configuration.md](configuration.md).


#### Step 3. Configure models

The VLM model is defined in:

```
vlm_delivery/input/model.json
```

You can directly swap in models supported by **OpenRouter** or **OpenAI** (e.g., gpt-4o, gpt-4.1, llama-3.1, etc.).

Just replace the model name and corresponding API key fields.

Before running, export your API key in the shell you will launch Jupyter from (e.g., `export OPENROUTER_API_KEY=...` or `export OPENAI_API_KEY=...`).

#### Step 4. Run the DeliveryBench evaluation

Open the quick-start notebook:

```
vlm_delivery/scripts/run_deliverybench.ipynb
```

This notebook will:

* connect to the UE server
* spawn courier agents
* run delivery episodes
* log and visualize results

#### Step 5. Analyze results

After runs finish, JSON result files will be exported to the directory specified by `lifecycle.export_path` in `vlm_delivery/input/experiment_config.json`. You can then aggregate them into CSV summaries using:

```bash
python vlm_delivery/evaluation/agent_performance_analysis.py \
  /path/to/result_json_folder \
  -o /path/to/output_dir
```

* `/path/to/result_json_folder` should point to a directory containing one or more JSON result files.
* The script will automatically load **all JSON files in the folder**, compute aggregate statistics (e.g., per-model averages), and write the CSV reports into `/path/to/output_dir`.

## 🧩 Advanced Usage

### Create a Custom DeliveryBench Map

DeliveryBench maps under `maps/<map_name>/` are generated in three stages. You can create your own city map assets (and the corresponding UE city environment) for training or evaluation by following the steps below.

#### Step 1. City Layout Generation

Generate a new procedural city layout (roads + buildings) and export the raw map assets under `maps/<map_name>/`.

```bash
python city_generation/generate_city_layout.py \
  --map-name <map_name> \
  --num-segments 35 \
  --seed 42
```

This will create the following files in `maps/<map_name>/`: `roads.json`, `buildings.json`, `elements.json`, `routes.json` (routes / bus routes), and `progen_world.json` (UE-compatible world objects).

#### Step 2. Enrich the City for DeliveryBench (POIs + gameplay metadata)

Take the base layout from Step 1 and add DeliveryBench-specific annotations (POI tags, bus routes, chargers). This produces the DeliveryBench-ready map file.

```bash
python city_generation/enrich_deliverybench_map.py \
  --map-dir maps/<map_name> \
  --seed 42
```

This will generate `maps/<map_name>/progen_world_enriched.json`, the DeliveryBench-ready map file. It enriches the base layout with DeliveryBench-specific annotations, including POI tags (e.g., `restaurant`, `store`, roadside chargers) and bus routes (`bus_routes`).

Optional sanity check (headless): render the generated map to a PNG and quickly verify the layout/POIs/bus routes look correct:

```bash
python vlm_delivery/scripts/test_map.py --map-name <map_name> --out-global /path/to/map.png
```

If `--out-global` is not provided, the image will be saved under `outputs/map_debug/` by default.

#### Step 3. UE city generation

After obtaining `progen_world.json`, you can generate the environment in UE. DeliveryBench leverages [SimWorld](https://github.com/SimWorld-AI/SimWorld)'s generation functionality. Please refer to the [world_generation](https://github.com/SimWorld-AI/SimWorld/blob/main/examples/world_generation.ipynb) for detailed instructions on generating the world in UE.

### Evaluating Custom Models

We also support plugging in **local VLMs**.
As a reference, we provide a minimal implementation of **LLaVA-OneVision** in:

```
vlm_delivery/vlm/base_model.py
```

You can adapt this file to wrap your own local model (e.g., by following the same `forward` / `generate` interface and image/text preprocessing pipeline).

The lightweight agentic workflow (including chain-of-thought reasoning and future planning) is implemented through:

* `vlm_delivery/gameplay/prompt.py` — prompt templates for actions, CoT, and future plans
* `vlm_delivery/gameplay/action_space.py` — parsing model outputs into structured actions
* `vlm_delivery/utils/vlm_prompt.py` — runtime prompt assembly (e.g., feeding the previous plan, observations, or action history back into the model)

You are free to extend this workflow with additional modules, such as:

* **memory modules** (e.g., episodic or long-term memory over past orders and routes)
* **reflection / self-correction loops** (e.g., asking the model to critique or refine its own plan)
* **tool-use modules** (e.g., calling external routing APIs or heuristic planners before acting)

### Running Multiple Environments in Parallel (Multi-Port)

To run multiple DeliveryBench instances in parallel, launch multiple SimWorld UE servers on different ports.

For each instance, edit the port in the extracted UE server package at:

```
gym_citynav/Binaries/Linux/unrealcv.ini
```

Then set the matching port in `vlm_delivery/input/experiment_config.json`:

* `gym_env.ue_port`: Must match the UE server port for that instance

Once each server uses a unique port (e.g., 9000, 9001, 9002, ...), you can run multiple experiments concurrently (one per port).


## 🧑‍💻For Contributors
We welcome contributions from the community! Whether you want to report bugs, suggest features, or submit code improvements, your input is valuable. Please check out our [Contributing Guidelines](contributing.md) for details on how to get started.