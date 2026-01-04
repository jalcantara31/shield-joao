[![maestro](https://github.com/lps-ufrj-br/maestro-lightning/actions/workflows/flow.yml/badge.svg)](https://github.com/lps-ufrj-br/maestro-lighning/actions/workflows/flow.yml)
[![pages](https://github.com/petrobras-lps-ufrj-br/barrier/actions/workflows/pages.yml/badge.svg)](https://github.com/petrobras-lps-ufrj-br/shield/actions/workflows/pages.yml)

# 🛡️ SHIELD: Dynamic Barrier Management

SHIELD is an advanced, data-driven framework designed to manage and monitor safety barriers dynamically in high-stakes industrial environments. By integrating real-time Emergency Shutdown (ESD) event data, SHIELD provides a continuous overview of operational integrity for both onshore and offshore assets.

## 📌 Overview
In complex energy and industrial operations, safety barriers are rarely static. SHIELD (Safety Hierarchy & Integrated Event Logic Dashboard) bridges the gap between theoretical safety studies and real-time operational reality.

The repository provides tools to visualize, analyze, and automate the status of safety barriers—whether physical, hardware, or procedural—ensuring that the "Swiss Cheese Model" of risk remains closed at all times.

## 🚀 Key Objectives
Dynamic Risk Assessment: Move beyond static spreadsheets to a live view of barrier health based on real-time ESD triggers.
Dual-Environment Support: Tailored logic for the unique challenges of Offshore (subsea, platforms, FPSOs) and Onshore (refineries, pipelines, plants) operations.
ESD Event Correlation: Automatically map Emergency Shutdown signals to specific barrier degradations.
Decision Support: Provide clear, actionable insights for O&M (Operations & Maintenance) teams during critical safety events.

## 🛠️ Core Features
Real-time Barrier Tracking: Monitors the status of Final Control Elements (FCE), Logic Solvers, and Sensors.
Automated Degradation Logic: Calculates the impact of bypassed or failed components on the overall safety integrity.
Geographic Visualization: Specialized mapping for onshore plots and offshore topside/subsea layouts.


## Installation:

### Installation:

```
make
```

### Launch a Jupyter notebook:

```
make jupyter
```

## Examples:

* [How to access Cognite database](examples/How_to_connect_to_cognite.ipynb)