from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    values: List[List[float]] = Field(..., description="2D numeric samples")
    methods: Optional[List[Literal["z_score", "isolation_forest", "lof", "hybrid"]]] = None
    z_threshold: float = 2.0
    contamination: float = 0.08
    n_estimators: int = 300
    lof_neighbors: int = 20


class CleanRequest(BaseModel):
    values: List[List[float]]
    methods: Optional[List[Literal["z_score", "isolation_forest", "lof", "hybrid"]]] = None
    z_threshold: float = 2.0
    contamination: float = 0.08
    n_estimators: int = 300
    lof_neighbors: int = 20


class LabelAnalyzeRequest(BaseModel):
    values: List[List[float]] = Field(..., description="2D numeric samples")
    labels: List[int] = Field(..., description="Class labels for each sample")
    k_neighbors: int = 5


class MethodInfo(BaseModel):
    name: str
    purpose: str


class HealthResponse(BaseModel):
    status: str
    service: str


class AnalyzeItem(BaseModel):
    index: int
    value: List[float]
    z_score_flag: bool
    isolation_forest_flag: bool
    lof_flag: bool
    final_flag: bool
    confidence: float


class AnalyzeSummary(BaseModel):
    total_samples: int
    suspicious_count: int
    anomaly_ratio: float
    enabled_methods: List[str]


class AnalyzeResponse(BaseModel):
    summary: AnalyzeSummary
    results: List[AnalyzeItem]
    suspicious_indices: List[int]
    cleaned_data: List[List[float]]


class CleanResponse(BaseModel):
    total_samples: int
    removed_samples: int
    cleaned_data: List[List[float]]
    suspicious_indices: List[int]


class LabelAnalyzeItem(BaseModel):
    index: int
    value: List[float]
    label: int
    suspicious: bool


class LabelAnalyzeResponse(BaseModel):
    total_samples: int
    suspicious_count: int
    suspicious_indices: List[int]
    results: List[LabelAnalyzeItem]