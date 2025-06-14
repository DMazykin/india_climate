# India District-Level Climate, Spatial, and Agricultural Data Database

## Overview

This project aims to create an analytical database for **district-level climate, spatial, and agricultural data in India**. The database is designed to support data analysis and visualization, with a focus on monthly updates to integrate new climate and agricultural data over time. Ultimately, this database will be linked to agricultural insurance data to enable risk assessment and trend analysis for agri-insurance at the district level.

### Key Components
- **Climate Data**: Monthly data on temperature, rainfall, and humidity.
- **Spatial Data**: District boundaries and geographic information.
- **Agricultural Data**: Crop types, yield, sowing/harvesting seasons, soil quality, and irrigation information.
- **Insurance Data** (Future Addition): Historical insurance claims, weather-based indices, and district-specific risk metrics.

## Tech Stack

The project primarily utilizes:
- **DuckDB**: For efficient storage, querying, and processing of large analytical datasets.
- **Ibis**: For flexible ETL (Extract, Transform, Load) operations and data manipulation.
- **Pydeck**: For visualizing spatial data and trends in the collected data.

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Poetry** for package and dependency management. If you don’t have Poetry installed, follow the [Poetry installation guide](https://python-poetry.org/docs/#installation) to set it up.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
