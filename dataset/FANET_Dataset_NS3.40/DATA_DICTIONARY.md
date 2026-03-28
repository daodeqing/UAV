# DATA_DICTIONARY

This document describes the structure and meaning of all variables contained in the dataset files.
Each table corresponds to one CSV file in the dataset.

---

## packet_trace.csv

| Column            | Type   | Unit | Description                                           |
|-------------------|--------|------|-------------------------------------------------------|
| TxTime_s          | float  | s    | Packet transmission start time                        |
| PacketUid         | int    |      | Unique packet identifier                              |
| NodeIdTx          | int    |      | ID of the transmitting UAV node                       |
| NodeIdDst         | int    |      | Intended destination node ID                          |
| TxPower_dBm       | float  | dBm  | Transmission power level                              |
| PacketSizeByte_tx | int    | bytes| Packet size transmitted                               |
| DataRate_Mbps     | float  | Mbps | Transmission data rate                                |
| RoleOfNodeTx      | string |      | Role of the transmitting node (e.g., source, relay)   |
| RxTime_s          | float  | s    | Reception time of the packet                          |
| NodeIdRx          | int    |      | ID of the receiving node                              |
| RSSI_dBm          | float  | dBm  | Received signal strength                              |
| Noise_dBm         | float  | dBm  | Measured noise power at receiver                      |
| SNR_dB            | float  | dB   | Signal-to-noise ratio                                 |
| RoleOfNodeRx      | string |      | Role of the receiving node (e.g., destination, relay) |
| Link_Delay_ms     | float  | ms   | Transmission delay between nodes                      |
| PacketTypePrimary | string |      | Primary packet type (e.g., HELLO, TC, DATA)           |
| PacketTypeSummary | string |      | Summary or aggregated packet classification           |

---

## network_qos_metrics.csv

| Column                   | Type  | Unit  | Description                                                 |
|---------------------------|-------|-------|-------------------------------------------------------------|
| window_start             | float | s     | Start time of the evaluation window                         |
| window_end               | float | s     | End time of the evaluation window                           |
| sent_pkts                | int   |       | Total packets sent during the window                        |
| sent_bytes               | int   | bytes | Total bytes sent                                            |
| recv_pkts                | int   |       | Total packets received                                      |
| recv_bytes               | int   | bytes | Total bytes received                                        |
| avg_delay_ms             | float | ms    | Average end-to-end delay                                    |
| jitter_ms                | float | ms    | Average packet delay variation                              |
| DestinationRecvDataPkts  | int   |       | Number of data packets successfully received at destination |
| DestinationRecvDataBytes | int   | bytes | Total data bytes received at destination                    |
| throughput_bps           | float | bps   | Network throughput                                          |
| goodput_bps              | float | bps   | Useful throughput excluding overhead                        |
| ETX                      | float |       | Expected transmission count metric                          |
| PDR                      | float |       | Packet delivery ratio                                       |
| LossRate                 | float |       | Packet loss rate                                            |

---

## node_qos_metrics.csv

| Column                   | Type  | Unit  | Description                                     |
|---------------------------|-------|-------|------------------------------------------------|
| window_start             | float | s     | Start time of the evaluation window             |
| window_end               | float | s     | End time of the evaluation window               |
| node_id                  | int   |       | Node identifier                                 |
| sent_pkts                | int   |       | Number of packets sent by the node              |
| sent_bytes               | int   | bytes | Total bytes sent by the node                    |
| recv_pkts                | int   |       | Number of packets received by the node          |
| recv_bytes               | int   | bytes | Total bytes received by the node                |
| avg_delay_ms             | float | ms    | Average delay of packets handled by the node    |
| jitter_ms                | float | ms    | Jitter at node level                            |
| throughput_bps           | float | bps   | Node throughput                                 |
| DestinationRecvDataPkts  | int   |       | Data packets successfully received at this node |
| DestinationRecvDataBytes | int   | bytes | Data bytes successfully received                |
| goodput_bps              | float | bps   | Node-level goodput excluding control overhead   |

---

## node_state.csv

| Column           | Type   | Unit | Description                               |
|------------------|--------|------|-------------------------------------------|
| time             | float  | s    | Simulation time                           |
| node_id          | int    |      | Node identifier                           |
| initial_energy   | float  | J    | Initial energy capacity                   |
| remaining_energy | float  | J    | Remaining energy at given time            |
| data_mode        | string |      | Data transmission mode (e.g., CBR, video) |
| data_rate_mbps   | float  | Mbps | Node data rate                            |
| pos_x            | float  | m    | X coordinate position                     |
| pos_y            | float  | m    | Y coordinate position                     |
| pos_z            | float  | m    | Altitude (Z coordinate)                   |
| vel_x            | float  | m/s  | Velocity component on X axis              |
| vel_y            | float  | m/s  | Velocity component on Y axis              |
| vel_z            | float  | m/s  | Velocity component on Z axis              |
| speed_m_s        | float  | m/s  | Instantaneous node speed                  |
| yaw_deg          | float  | °    | Yaw angle orientation                     |
| pitch_deg        | float  | °    | Pitch angle orientation                   |

---

## olsr_links.csv

| Column           | Type  | Unit | Description                           |
|------------------|-------|------|---------------------------------------|
| time             | float | s    | Simulation time                       |
| node_id          | int   |      | Node identifier                       |
| neighbor_id      | int   |      | Neighbor node ID                      |
| distance_m       | float | m    | Distance between nodes                |
| symmetrical_link | int   |      | 1 if link is symmetrical, 0 otherwise |

---

## olsr_node_state.csv

| Column                | Type  | Unit | Description                                       |
|-----------------------|-------|------|---------------------------------------------------|
| time                  | float | s    | Simulation time                                   |
| node_id               | int   |      | Node identifier                                   |
| is_mpr                | int   |      | 1 if node is MPR (Multipoint Relay), 0 otherwise  |
| one_hop_neighbors     | int   |      | Number of one-hop neighbors                       |
| sym_one_hop_neighbors | int   |      | Number of symmetrical one-hop neighbors           |
| two_hop_neighbors     | int   |      | Number of two-hop neighbors                       |
| sym_two_hop_neighbors | int   |      | Number of symmetrical two-hop neighbors           |
| mpr_set_size          | int   |      | Size of the node’s MPR set                        |
| mpr_selector_set_size | int   |      | Size of the MPR selector set                      |
| mpr_changes           | int   |      | Number of MPR changes over time                   |
| olsr_hello_sent       | int   |      | Number of HELLO packets sent                      |
| olsr_hello_received   | int   |      | Number of HELLO packets received                  |
| olsr_tc_generated     | int   |      | Number of topology control (TC) packets generated |
| olsr_tc_retransmitted | int   |      | Number of TC packets retransmitted                |
| olsr_tc_recved        | int   |      | Number of TC packets received                     |

---

## simulation_scenario.csv

| Column                | Type   | Unit | Description                                       |
|-----------------------|--------|------|---------------------------------------------------|
| scenario_id           | string |      | Scenario identifier (e.g., S1–S8)                 |
| node_density_level    | string |      | Density level (LOW, MEDIUM, HIGH)                 |
| node_speed_level      | string |      | Speed category of UAVs                            |
| energy_capacity_level | string |      | Energy level configuration                        |
| range_level           | string |      | Communication range category                      |
| traffic_type          | string |      | Type of traffic used (CBR or Video)               |
| communication_type    | string |      | Communication type (UAV↔UAV, UAV↔BS, or Mixed)    |
| num_source_nodes      | int    |      | Number of source UAVs                             |
| num_destination_nodes | int    |      | Number of destination UAVs                        |
| protocol              | string |      | Routing protocol used (OLSR)                      |
| total_time            | float  | s    | Total simulation duration                         |
| simulation_zone       | string |      | Simulation area dimensions (e.g., 1000×1000×1000 m) |

---

**Note:**
This dictionary provides a human-readable overview of the dataset structure.
For programmatic access, the same information is also available in `DATA_DICTIONARY.csv`.

