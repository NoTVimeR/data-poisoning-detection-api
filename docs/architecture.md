# Final Architecture of the Data Poisoning Detection System

```mermaid
flowchart TD
    A["Data Sources<br/>Synthetic + Breast Cancer Wisconsin"] --> B["Poisoning Simulation"]
    B --> B1["Simple Outlier Attack"]
    B --> B2["Complex Subtle Poisoning<br/>Correlation Drift"]
    B --> B3["Label Flipping Attack"]

    B1 --> C["FastAPI Detection Service v2"]
    B2 --> C
    B3 --> C

    C --> D["Preprocessing Layer<br/>Scaling + Matrix Formatting"]
    D --> E["Detection Layer"]

    E --> E1["Z-score"]
    E --> E2["Isolation Forest"]
    E --> E3["Local Outlier Factor"]
    E --> E4["KNN Label Consistency"]

    E1 --> F["Decision Engine"]
    E2 --> F
    E3 --> F
    E4 --> F

    F --> G["Hybrid Voting + Confidence Score"]
    G --> H["API Response"]

    H --> H1["Suspicious Indices"]
    H --> H2["Method Flags"]
    H --> H3["Cleaned Data"]
    H --> H4["Evaluation Metrics"]

    H3 --> I["Downstream ML Pipeline"]
    I --> J["Model Training on Filtered Data"]
```

The system works as an API-based security layer for machine learning pipelines. Input data is analyzed before model training, suspicious samples are flagged, and a cleaned dataset can be returned for downstream use.
