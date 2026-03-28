#  FANET Dataset: UAV Communication Scenarios in NS-3.40

## Brief Summary
A comprehensive dataset of eight FANET simulation scenarios in NS-3.40, providing detailed UAV mobility, energy, and QoS metrics for research on network behavior, optimization, and AI-based routing.

---

## 1. Description

This dataset provides detailed information about the **behavior of UAVs in Flying Ad hoc Networks (FANETs)** under different mobility, energy, and communication conditions.
It includes **eight distinct simulation scenarios**, each generated with the **NS-3.40 simulator**, using the **OLSR routing protocol** as a stable baseline.

Although OLSR is employed, **the objective of this dataset is not to evaluate OLSR performance**.
Instead, the goal is to offer a **rich benchmark describing UAV behavior and network dynamics** in diverse conditions.
Therefore, choosing OLSR — or any specific FANET routing protocol — is **not critical** for the purpose of this dataset.
The focus is on the **mobility, energy evolution, QoS, and connectivity patterns** observed in UAV networks.

---

## 2. Dataset Structure

Each scenario is stored in a separate folder:
The root directory contains the main documentation and metadata files.

FANET_Dataset_NS3.40/
│
├── README.md
├── DATA_DICTIONARY.md
├── DATA_DICTIONARY.csv
├── CITATION.cff
├── LICENSE.txt
│
├── Scenario_1/
│ ├── packet_trace.csv
│ ├── network_qos_metrics.csv
│ ├── node_qos_metrics.csv
│ ├── node_state.csv
│ ├── olsr_links.csv
│ ├── olsr_node_state.csv
│ └── simulation_scenario.csv
│
├── Scenario_2/
│ ├── packet_trace.csv
│ ├── network_qos_metrics.csv
│ ├── node_qos_metrics.csv
│ ├── node_state.csv
│ ├── olsr_links.csv
│ ├── olsr_node_state.csv
│ └── simulation_scenario.csv
│
├── Scenario_3/
│ ├── packet_trace.csv
│ ├── network_qos_metrics.csv
│ ├── node_qos_metrics.csv
│ ├── node_state.csv
│ ├── olsr_links.csv
│ ├── olsr_node_state.csv
│ └── simulation_scenario.csv
│
├── Scenario_4/
│ ├── packet_trace.csv
│ ├── network_qos_metrics.csv
│ ├── node_qos_metrics.csv
│ ├── node_state.csv
│ ├── olsr_links.csv
│ ├── olsr_node_state.csv
│ └── simulation_scenario.csv
│
├── Scenario_5/
│ ├── packet_trace.csv
│ ├── network_qos_metrics.csv
│ ├── node_qos_metrics.csv
│ ├── node_state.csv
│ ├── olsr_links.csv
│ ├── olsr_node_state.csv
│ └── simulation_scenario.csv
│
├── Scenario_6/
│ ├── packet_trace.csv
│ ├── network_qos_metrics.csv
│ ├── node_qos_metrics.csv
│ ├── node_state.csv
│ ├── olsr_links.csv
│ ├── olsr_node_state.csv
│ └── simulation_scenario.csv
│
├── Scenario_7/
│ ├── packet_trace.csv
│ ├── network_qos_metrics.csv
│ ├── node_qos_metrics.csv
│ ├── node_state.csv
│ ├── olsr_links.csv
│ ├── olsr_node_state.csv
│ └── simulation_scenario.csv
│
└── Scenario_8/
├── packet_trace.csv
├── network_qos_metrics.csv
├── node_qos_metrics.csv
├── node_state.csv
├── olsr_links.csv
├── olsr_node_state.csv
└── simulation_scenario.csv

---

## 3. Files and Content

Each scenario folder contains the following files:

| File name                 | Description                                                                                             |
|------------               |-------------------------------------------------------------------------------------------------------  |
| `packet_trace.csv`        | Contains transmission and reception events for each packet, including times, node IDs, signal strength (RSSI), SNR, and delays. |
| `network_qos_metrics.csv` | Network-wide QoS metrics (delay, jitter, throughput, goodput, ETX, PDR, loss rate) per time window.     |
| `node_qos_metrics.csv`    | Node-level QoS metrics (sent/received packets, throughput, delay, jitter).                              |
| `node_state.csv`          | UAV energy levels, position, velocity, orientation, and movement parameters over time.                  |
| `olsr_links.csv`          | Link states between nodes (distance, link symmetry).                                                    |
| `olsr_node_state.csv`     | OLSR-related variables such as neighbor sets, MPR sets, and message counts.                             |
| `simulation_scenario.csv` | General metadata about each scenario (node density, mobility, energy range, communication type, etc.).  |

---

## 4. Scenarios Overview

Each scenario directory (from S1 to S8) includes its own `simulation_scenario.csv` file.
This file describes the configuration parameters of the corresponding simulation, including node density, speed, 
energy capacity, transmission range, traffic type, communication type, and other relevant settings.
This structure ensures that each scenario is self-contained and can be independently analyzed or reproduced.

### Parameter Levels

| Parameter              | Level   | Description / Interval |
|------------            |-------- |------------------------|
| **Node Density**       | Low     | 11 nodes               |
|                        | Medium  | 31 nodes               |
|                        | High    | 51 nodes               |
| **Speed**              | Low     | 10 m/s                 |
|                        | Medium  | 15 m/s                 |
|                        | High    | 20 m/s                 |
| **Initial Energy**     | Low     | [100 – 150] J          |
|                        | Medium  | [150 – 200] J          |
|                        | High    | [200 – 300] J          |
| **Transmission Range** | Low     | 250 m                  |
|                        | Medium  | 350 m                  |
|                        | High    | 400 m                  |
| **Traffic Type**       | —       | CBR / Video            |

---

### Scenario Configuration Summary

| Scenario  | Architecture             | Node Density | Speed  | Energy | Range  | Traffic Type |
|-----------|--------------            |--------------|--------|--------|------- |--------------|
| S1        | UAV ↔ UAV                | Low          | Low    | Low    | High   | CBR          |
| S2        | Mixed (UAV↔UAV + UAV↔BS) | Low          | Medium | Medium | High   | Video        |
| S3        | UAV ↔ Base Station       | Medium       | Medium | Medium | Medium | CBR          |
| S4        | Mixed (UAV↔UAV + UAV↔BS) | Medium       | Low    | Low    | Low    | CBR          |
| S5        | UAV ↔ UAV                | Medium       | High   | High   | Medium | Video        |
| S6        | UAV ↔ UAV                | High         | Low    | Medium | Low    | CBR          |
| S7        | Mixed (UAV↔UAV + UAV↔BS) | High         | Medium | High   | Low    | Video        |
| S8        | UAV ↔ Base Station       | High         | High   | Low    | Low    | CBR          |

---

##  5. Simulation Environment

- **Simulator:** NS-3.40
- **Routing protocol:** OLSR (standard implementation) 
- **Simulation area:** 1000 × 1000 × 200 m
- **Mobility model:** Random waypoint (3D)
- **Energy model:** BasicEnergySource + WifiRadioEnergyModel
- **Metrics recorded:** QoS, link state, node state, and energy consumption

---

## 6. Usage Recommendations

This dataset can be used for:

- **Analyzing FANET network dynamics**, including topology evolution, link stability, and connectivity patterns.
- **Evaluating Quality of Service (QoS)** and **energy consumption** in multi-UAV communication systems.
- **Designing and validating intelligent routing algorithms**, such as AI or machine learning–based approaches.
- **Testing energy-aware or mobility-aware communication strategies** in 3D UAV environments.
- **Comparing performance** across different network configurations and operational conditions.
- **Training predictive models** for link quality estimation, UAV positioning, or adaptive power management.

This dataset is particularly useful for researchers working on FANETs, MANETs, IoT-based UAV communication, and AI-driven networking.

---

##  7. Data Dictionary

For detailed variable descriptions and units:
- `DATA_DICTIONARY.md` → Human-readable version (Markdown format)
- `DATA_DICTIONARY.csv` → Machine-readable version (for Python, R, etc.)

Both files describe all variables for each CSV file in the dataset.

---

##  8. Citation

If you use this dataset, please cite it as:

> **Ali, MOUSSAOUI. (2025). FANET Dataset: UAV Communication Scenarios in NS-3.40.** Zenodo.
> DOI: *(to be provided after publication)*

##  9. LICENSE
This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
You are free to share and adapt the material, provided that proper credit is given to the original author.

##  10. Contact

For any questions, collaboration requests, or data issues, please contact:

**Ali MOUSSAOUI**
Department of Computer Science, Intelligent Systems and Cognitive Computing (ISCC) Laboratory
University Mohamed El Bachir El Ibrahimi, Bordj Bou Arréridj, Algeria
 Email: ali.moussaoui@univ-bba.dz 

