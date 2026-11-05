/** Shared RAG mode labels for manual chat + CSV eval UIs (must match backend `rag` | `no_rag` | `both`). */

export const RAG_MODE_UI = ["RAG'li", "RAG'siz", 'İkisi birden'] as const

export type RagModeUiLabel = (typeof RAG_MODE_UI)[number]

export type RagModeApi = 'rag' | 'no_rag' | 'both'

export const ragModeToApi: Record<RagModeUiLabel, RagModeApi> = {
  "RAG'li": 'rag',
  "RAG'siz": 'no_rag',
  'İkisi birden': 'both',
}
