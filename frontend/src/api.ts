import type {
  AgentInvestigationResponse,
  ExplainResponse,
  FairnessResponse,
  HealthResponse,
  InvestigationResponse,
  MetricsResponse,
  ModelComparisonResponse,
  PredictionRequest,
  PredictionResponse,
  SecurityEventsResponse,
  SimilarCasesResponse,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "/api";

export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(status: number, payload: unknown) {
    super(
      typeof payload === "object" &&
        payload !== null &&
        "detail" in payload &&
        typeof payload.detail === "string"
        ? payload.detail
        : `Request failed with status ${status}`,
    );

    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new ApiError(response.status, payload);
  }

  return payload as T;
}

export function predict(
  transaction: PredictionRequest,
): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(transaction),
  });
}

export function explain(
  transaction: PredictionRequest,
): Promise<ExplainResponse> {
  return request<ExplainResponse>("/explain", {
    method: "POST",
    body: JSON.stringify(transaction),
  });
}

export function similarCases(
  description: string,
  topK = 5,
): Promise<SimilarCasesResponse> {
  return request<SimilarCasesResponse>("/similar-cases", {
    method: "POST",
    body: JSON.stringify({
      description,
      top_k: topK,
    }),
  });
}

export function investigate(
  question: string,
  transactionDescription: string,
): Promise<InvestigationResponse> {
  return request<InvestigationResponse>("/investigate", {
    method: "POST",
    body: JSON.stringify({
      question,
      transaction_description: transactionDescription,
    }),
  });
}

export function agentInvestigate(
  question: string,
  transactionDescription: string,
  transaction: PredictionRequest,
): Promise<AgentInvestigationResponse> {
  return request<AgentInvestigationResponse>("/agent-investigate", {
    method: "POST",
    body: JSON.stringify({
      question,
      transaction_description: transactionDescription,
      transaction,
    }),
  });
}

export function getAgentCase(
  caseId: string,
): Promise<AgentInvestigationResponse> {
  return request<AgentInvestigationResponse>(
    `/agent-investigate/${encodeURIComponent(caseId)}`,
  );
}

export function health(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function metrics(): Promise<MetricsResponse> {
  return request<MetricsResponse>("/metrics");
}

export function fairness(): Promise<FairnessResponse> {
  return request<FairnessResponse>("/fairness");
}

export function modelComparison(): Promise<ModelComparisonResponse> {
  return request<ModelComparisonResponse>("/model-comparison");
}

export function securityEvents(): Promise<SecurityEventsResponse> {
  return request<SecurityEventsResponse>("/security-events");
}