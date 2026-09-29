export interface PredictionRequest {
  hour_of_day: number;
  is_weekend: number;
  is_night_transaction: number;
  country: string;
  city: string;
  merchant_category: string;
  payment_method: string;
  device_type: string;
  customer_age: number;
  credit_score: number;
  account_age_years: number;
  account_balance: number;
  transaction_amount: number;
  num_prev_transactions: number;
  transaction_freq_monthly: number;
  distance_from_home_km: number;
  time_since_last_txn_hrs: number;
  is_international: number;
  failed_attempts: number;
  pin_changed_recently: number;
}

export interface PredictionResponse {
  prediction: "Fraudulent" | "Legitimate";
  fraud_probability: number;
  risk_level: "Low" | "Medium" | "High";
}

export interface FeatureContribution {
  feature: string;
  feature_value: string;
  shap_value: number;
  direction: "higher" | "lower";
}

export interface ExplainResponse {
  prediction: "Fraudulent" | "Legitimate";
  fraud_probability: number;
  top_contributing_features: FeatureContribution[];
}

export interface SimilarCaseResult {
  case_id: string;
  fraud_type: string;
  key_observations: string;
  outcome: string;
  similarity_score: number;
}

export interface SimilarCasesResponse {
  cases: SimilarCaseResult[];
}

export interface InvestigationSource {
  document_id: string;
  title: string;
  document_type: string;
  excerpt: string;
  similarity_score: number;
}

export interface InvestigationResponse {
  investigation_response: string;
  observed_evidence: string[];
  relevant_context: string[];
  recommended_next_steps: string[];
  sources: InvestigationSource[];
  evidence_insufficient: boolean;
  generation_mode: "llm" | "fallback";
}

export interface AgentInvestigationResponse {
  case_id: string;
  route_taken: "direct_assessment" | "multi_agent";
  agent_findings: Record<string, string>;
  tools_used: string[];
  final_investigation_summary: string;
  sources: Array<Record<string, unknown>>;
  evidence: string[];
  tool_failures: string[];
}

export interface HealthResponse {
  status: "healthy" | "unhealthy";
  model_loaded: boolean;
}

export interface MetricsResponse {
  prediction_request_count: number;
  error_count: number;
  average_request_latency_ms: number;
  last_request_latency_ms: number | null;
}

export interface SecurityEvent {
  action: "blocked" | "sanitized";
  category:
    | "prompt_injection"
    | "jailbreak"
    | "sensitive_data_request"
    | "unsafe_content";
  endpoint: string;
}

export interface SecurityEventsResponse {
  events: SecurityEvent[];
}

export interface FairnessGroupMetric {
  group: string;
  sample_count: number;
  actual_fraud_rate: number;
  predicted_fraud_rate: number;
  true_positive_rate: number | null;
  false_positive_rate: number | null;
}

export interface FairnessAttributeSummary {
  selection_rate_gap: number;
  groups: FairnessGroupMetric[];
}

export interface FairnessResponse {
  evaluation_rows: number;
  groups_checked: string[];
  overall_predicted_fraud_rate: number;
  summaries: Record<string, FairnessAttributeSummary>;
  limitations: string[];
}

export interface ModelComparison {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  inference_time_seconds: number;
  epochs_trained: number | null;
}

export interface ModelComparisonResponse {
  dataset: string;
  random_state: number;
  models: ModelComparison[];
  best_by_f1: {
    model: string;
    f1: number;
  };
  best_by_roc_auc: {
    model: string;
    roc_auc: number;
  };
}

export interface SecurityBlockedResponse {
  blocked: true;
  category:
    | "prompt_injection"
    | "jailbreak"
    | "sensitive_data_request"
    | "unsafe_content";
  message: string;
}