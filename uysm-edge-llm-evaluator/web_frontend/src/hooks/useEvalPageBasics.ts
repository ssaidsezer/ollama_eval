import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { SYSTEM_EVAL_PROFILE_ID } from '../eval/profileConstants'
import type { ModelProfile } from '../pages/ModelProfilesPage'
import { ragTypeToFlags, type RagTypeUi } from '../utils/collections'
import { ragModeToApi, type RagModeApi, type RagModeUiLabel } from '../eval/evalRagMode'

export type EvalBackendChoice = 'DeepSeek' | 'OpenAI' | 'Yerel (Ollama)'

export function useEvalPageBasics() {
  const ollamaQ = useQuery({
    queryKey: ['ollama-models'],
    queryFn: async () => (await api.get('/api/models/ollama')).data,
  })
  const embedQ = useQuery({
    queryKey: ['embed-models'],
    queryFn: async () => (await api.get('/api/models/embeddings')).data,
  })
  const cfg = useQuery({
    queryKey: ['config'],
    queryFn: async () => (await api.get('/api/config')).data,
  })
  const profilesQ = useQuery({
    queryKey: ['model-profiles'],
    queryFn: async () => (await api.get<{ profiles: ModelProfile[] }>('/api/model-profiles')).data,
  })

  const allModels = ollamaQ.data?.models ?? []
  const embedModels = embedQ.data?.models ?? []

  const [qaSelected, setQaSelected] = useState<string[]>([])
  const [customModels, setCustomModels] = useState<string[]>([])

  const [evalEnabled, setEvalEnabled] = useState(true)
  const [evalBackend, setEvalBackend] = useState<EvalBackendChoice>('DeepSeek')
  const [evalModelName, setEvalModelName] = useState('')
  const [localEvalModel, setLocalEvalModel] = useState('')

  const [embedModel, setEmbedModel] = useState('')
  const [ragType, setRagType] = useState<RagTypeUi>('Klasik')
  const [collectionName, setCollectionName] = useState('')

  const [ragModeUi, setRagModeUi] = useState<RagModeUiLabel>("RAG'li")
  const [k, setK] = useState(5)
  const [scoreTh, setScoreTh] = useState(0.55)

  const [openaiKey, setOpenaiKey] = useState('')

  const [useSavedQaDefaults, setUseSavedQaDefaults] = useState(true)
  const [qaProfileByModel, setQaProfileByModel] = useState<Record<string, string>>({})
  const [evalProfileId, setEvalProfileId] = useState(SYSTEM_EVAL_PROFILE_ID)
  const useSavedEvalDefaults = evalProfileId !== SYSTEM_EVAL_PROFILE_ID
  const [thinkingEnabled, setThinkingEnabled] = useState(false)

  useEffect(() => {
    const providerDefault =
      evalBackend === 'OpenAI'
        ? cfg.data?.openai_eval_model_name
        : cfg.data?.deepseek_eval_model_name || cfg.data?.eval_model_name
    if ((evalBackend === 'DeepSeek' || evalBackend === 'OpenAI') && providerDefault) {
      setEvalModelName(providerDefault)
    }
  }, [cfg.data, evalBackend])

  useEffect(() => {
    if (embedModels.length && !embedModel) setEmbedModel(embedModels[0])
  }, [embedModels, embedModel])

  useEffect(() => {
    if (allModels.length && !localEvalModel) setLocalEvalModel(allModels[0])
  }, [allModels, localEvalModel])

  useEffect(() => {
    setQaProfileByModel((prev) => {
      const pruned: Record<string, string> = {}
      for (const m of qaSelected) if (prev[m]) pruned[m] = prev[m]
      return pruned
    })
  }, [qaSelected])

  const setQaProfileForModel = (model: string, profileId: string) => {
    setQaProfileByModel((prev) => {
      if (!profileId) {
        if (!prev[model]) return prev
        const next = { ...prev }
        delete next[model]
        return next
      }
      if (prev[model] === profileId) return prev
      return { ...prev, [model]: profileId }
    })
  }

  const { smartRag, retrievalMode } = ragTypeToFlags(ragType)
  const ragModeApi: RagModeApi = ragModeToApi[ragModeUi]

  const qaProfiles = profilesQ.data?.profiles ?? []
  const evalProfiles = (profilesQ.data?.profiles ?? []).filter((p) => p.model_name === 'eval')

  useEffect(() => {
    if (evalProfileId === SYSTEM_EVAL_PROFILE_ID) return
    if (!evalProfiles.length) {
      setEvalProfileId(SYSTEM_EVAL_PROFILE_ID)
      return
    }
    if (evalProfiles.some((p) => p.id === evalProfileId)) return
    setEvalProfileId(SYSTEM_EVAL_PROFILE_ID)
  }, [evalProfiles, evalProfileId])

  return {
    ollamaQ,
    embedQ,
    cfg,
    profilesQ,
    allModels,
    embedModels,
    qaSelected,
    setQaSelected,
    customModels,
    setCustomModels,
    evalEnabled,
    setEvalEnabled,
    evalBackend,
    setEvalBackend,
    evalModelName,
    setEvalModelName,
    localEvalModel,
    setLocalEvalModel,
    embedModel,
    setEmbedModel,
    ragType,
    setRagType,
    collectionName,
    setCollectionName,
    ragModeUi,
    setRagModeUi,
    k,
    setK,
    scoreTh,
    setScoreTh,
    openaiKey,
    setOpenaiKey,
    useSavedQaDefaults,
    setUseSavedQaDefaults,
    qaProfileByModel,
    setQaProfileByModel,
    setQaProfileForModel,
    useSavedEvalDefaults,
    evalProfileId,
    setEvalProfileId,
    thinkingEnabled,
    setThinkingEnabled,
    smartRag,
    retrievalMode,
    ragModeApi,
    qaProfiles,
    evalProfiles,
  }
}
