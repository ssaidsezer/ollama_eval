import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import CircularProgress from '@mui/material/CircularProgress'
import Divider from '@mui/material/Divider'
import FormControlLabel from '@mui/material/FormControlLabel'
import MenuItem from '@mui/material/MenuItem'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'

export type ModelProfile = {
  id: string
  name: string
  model_name: string
  system_prompt: string | null
  params: Record<string, unknown>
  is_default: boolean
  version: number
  updated_at: string
}

type OllamaModelsPayload = {
  models: string[]
  error?: string | null
  filter_elapsed_seconds?: number
  filtered_embedding_count?: number
}

const emptyParams = {
  num_ctx: '',
  temperature: '',
  top_p: '',
  repeat_penalty: '',
  num_predict: '',
  think: false as boolean,
}

export function ModelProfilesPage() {
  const qc = useQueryClient()
  const listQ = useQuery({
    queryKey: ['model-profiles'],
    queryFn: async () => (await api.get<{ profiles: ModelProfile[] }>('/api/model-profiles')).data,
  })
  const ollamaQ = useQuery({
    queryKey: ['ollama-models'],
    queryFn: async () => (await api.get<OllamaModelsPayload>('/api/models/ollama')).data,
  })
  const profiles = listQ.data?.profiles ?? []
  const allModels = ollamaQ.data?.models ?? []
  const modelSet = useMemo(() => new Set(allModels), [allModels])

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [modelSearch, setModelSearch] = useState('')

  const [draftName, setDraftName] = useState('')
  const [draftModel, setDraftModel] = useState('')
  const [draftPrompt, setDraftPrompt] = useState('')
  const [draftDefault, setDraftDefault] = useState(false)
  const [draftParams, setDraftParams] = useState({ ...emptyParams })
  const [showDraftForm, setShowDraftForm] = useState(false)

  const selected = useMemo(
    () => profiles.find((p) => p.id === selectedId) ?? null,
    [profiles, selectedId],
  )

  const resetDraft = (modelName = '') => {
    setDraftName('Yeni profil')
    setDraftModel(modelName)
    setDraftPrompt('')
    setDraftDefault(false)
    setDraftParams({ ...emptyParams })
  }

  useEffect(() => {
    if (!selected) return
    setShowDraftForm(true)
    setDraftName(selected.name)
    setDraftModel(selected.model_name)
    setDraftPrompt(selected.system_prompt ?? '')
    setDraftDefault(selected.is_default)
    const p = selected.params || {}
    setDraftParams({
      num_ctx: p.num_ctx != null ? String(p.num_ctx) : '',
      temperature: p.temperature != null ? String(p.temperature) : '',
      top_p: p.top_p != null ? String(p.top_p) : '',
      repeat_penalty: p.repeat_penalty != null ? String(p.repeat_penalty) : '',
      num_predict: p.num_predict != null ? String(p.num_predict) : '',
      think: Boolean(p.think),
    })
  }, [selected])

  const filteredOllamaModels = useMemo(() => {
    const q = modelSearch.trim().toLowerCase()
    if (!q) return allModels
    return allModels.filter((m) => m.toLowerCase().includes(q))
  }, [allModels, modelSearch])

  const modelProfiles = useMemo(() => {
    const m = draftModel.trim()
    if (!m) return []
    return profiles.filter((p) => p.model_name === m)
  }, [profiles, draftModel])

  /** İlk yanıt geldikten sonra true; arka plandaki yenilemede Kaydet’i kapatmaz */
  const ollamaFetched = ollamaQ.isFetched
  const canPersistProfile = draftName.trim() !== '' && draftModel.trim() !== ''

  const buildParamsPayload = (): Record<string, unknown> => {
    const out: Record<string, unknown> = {}
    if (draftParams.num_ctx !== '') out.num_ctx = parseInt(draftParams.num_ctx, 10)
    if (draftParams.temperature !== '') out.temperature = parseFloat(draftParams.temperature)
    if (draftParams.top_p !== '') out.top_p = parseFloat(draftParams.top_p)
    if (draftParams.repeat_penalty !== '') out.repeat_penalty = parseFloat(draftParams.repeat_penalty)
    if (draftParams.num_predict !== '') out.num_predict = parseInt(draftParams.num_predict, 10)
    if (draftParams.think) out.think = true
    return out
  }

  const createMut = useMutation({
    mutationFn: async () => {
      const { data } = await api.post<ModelProfile>('/api/model-profiles', {
        name: draftName.trim(),
        model_name: draftModel.trim(),
        system_prompt: draftPrompt.trim() || null,
        params: buildParamsPayload(),
        is_default: draftDefault,
      })
      return data
    },
    onSuccess: (p) => {
      void qc.invalidateQueries({ queryKey: ['model-profiles'] })
      setSelectedId(p.id)
    },
  })

  const saveMut = useMutation({
    mutationFn: async () => {
      if (!selectedId) throw new Error('Profil seçilmedi')
      const { data } = await api.put<ModelProfile>(`/api/model-profiles/${selectedId}`, {
        name: draftName.trim(),
        model_name: draftModel.trim(),
        system_prompt: draftPrompt.trim() || null,
        system_prompt_clear: false,
        params: buildParamsPayload(),
        is_default: draftDefault,
      })
      return data
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['model-profiles'] }),
  })

  const deleteMut = useMutation({
    mutationFn: async (id: string) => {
      await api.delete(`/api/model-profiles/${id}`)
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['model-profiles'] })
      setSelectedId(null)
      setShowDraftForm(false)
      resetDraft(draftModel)
    },
  })

  const cloneMut = useMutation({
    mutationFn: async (id: string) => {
      const { data } = await api.post<ModelProfile>(`/api/model-profiles/${id}/clone`)
      return data
    },
    onSuccess: (p) => {
      void qc.invalidateQueries({ queryKey: ['model-profiles'] })
      setSelectedId(p.id)
    },
  })

  const errMsg =
    (createMut.error as any)?.response?.data?.detail ||
    (saveMut.error as any)?.response?.data?.detail ||
    (deleteMut.error as any)?.response?.data?.detail ||
    (cloneMut.error as any)?.response?.data?.detail

  const isEvalModel = draftModel === 'eval'
  const orphanedModel =
    !isEvalModel &&
    selectedId &&
    draftModel.trim() !== '' &&
    ollamaFetched &&
    allModels.length > 0 &&
    !modelSet.has(draftModel.trim())

  return (
    <Stack direction={{ xs: 'column', lg: 'row' }} spacing={2} sx={{ alignItems: 'flex-start' }}>
      <Box sx={{ width: { xs: '100%', lg: 260 }, flexShrink: 0 }}>
        <Typography variant="h6" gutterBottom>
          Yerel modeller (Ollama)
        </Typography>
        {ollamaQ.data?.error && (
          <Alert severity="error" sx={{ mb: 1 }}>
            {String(ollamaQ.data.error)}
          </Alert>
        )}
        {!ollamaFetched && (
          <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 1, mb: 1 }}>
            <CircularProgress size={18} />
            <Typography variant="body2" color="text.secondary">
              Modeller yükleniyor…
            </Typography>
          </Box>
        )}
        <TextField
          size="small"
          fullWidth
          placeholder="Model ara..."
          value={modelSearch}
          onChange={(e) => setModelSearch(e.target.value)}
          disabled={!allModels.length && !!ollamaQ.data?.error}
          sx={{ mb: 1 }}
        />
        <Box
          sx={{
            maxHeight: 420,
            overflow: 'auto',
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 1,
          }}
        >
          {filteredOllamaModels.map((m) => (
            <Box
              key={m}
              onClick={() => {
                setSelectedId(null)
                resetDraft(m)
                setShowDraftForm(false)
              }}
              sx={{
                p: 1,
                cursor: 'pointer',
                bgcolor: m === draftModel ? 'action.selected' : 'transparent',
                '&:hover': { bgcolor: 'action.hover' },
              }}
            >
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                {m}
              </Typography>
            </Box>
          ))}
          {ollamaFetched && allModels.length === 0 && !ollamaQ.data?.error && (
            <Typography variant="body2" sx={{ p: 2 }} color="text.secondary">
              Yerel model bulunamadı. Ollama çalışıyor ve modeller yüklü mü kontrol edin.
            </Typography>
          )}
          {ollamaFetched && allModels.length > 0 && filteredOllamaModels.length === 0 && (
            <Typography variant="body2" sx={{ p: 2 }} color="text.secondary">
              Aramaya uygun model yok.
            </Typography>
          )}
        </Box>

        <Divider sx={{ my: 1.5 }} />
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5, px: 0.5 }}>
          Eval modeli
        </Typography>
        <Box
          onClick={() => {
            setSelectedId(null)
            resetDraft('eval')
            setShowDraftForm(true)
          }}
          sx={{
            p: 1,
            cursor: 'pointer',
            borderRadius: 1,
            bgcolor: draftModel === 'eval' ? 'action.selected' : 'transparent',
            '&:hover': { bgcolor: 'action.hover' },
          }}
        >
          <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
            Eval modeli
          </Typography>
        </Box>
      </Box>

      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography variant="h6" gutterBottom>
          {selectedId ? 'Profili düzenle' : 'Yeni profil oluştur'}
        </Typography>
        {errMsg && (
          <Alert severity="error" sx={{ mb: 1 }}>
            {String(errMsg)}
          </Alert>
        )}
        {orphanedModel && (
          <Alert severity="info" sx={{ mb: 1 }}>
            Bu model yerel Ollama listesinde yok — OpenAI/harici model profili olarak kaydedilecek.
          </Alert>
        )}
        <Stack spacing={2}>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'minmax(280px, 1fr) auto' },
              gap: 1.5,
              alignItems: 'center',
            }}
          >
            <TextField
              select
              size="small"
              label="Bu modele ait kayıtlı profil"
              value={selectedId ?? ''}
              onChange={(e) => {
                const id = e.target.value
                if (!id) {
                  setSelectedId(null)
                  resetDraft(draftModel)
                  setShowDraftForm(false)
                  return
                }
                setSelectedId(id)
                setShowDraftForm(true)
              }}
              disabled={!draftModel.trim() || modelProfiles.length === 0}
              fullWidth
            >
              {modelProfiles.map((p) => (
                <MenuItem key={p.id} value={p.id}>
                  {p.name}
                  {p.is_default ? ' · varsayılan' : ''}
                </MenuItem>
              ))}
            </TextField>
            <Button
              variant="outlined"
              size="small"
              onClick={() => {
                setSelectedId(null)
                resetDraft(draftModel)
                setShowDraftForm(true)
              }}
            >
              Yeni profil
            </Button>
          </Box>
          {!showDraftForm && !selectedId && (
            <Typography variant="body2" color="text.secondary">
              Önce bu model için kayıtlı bir profil seçin ya da Yeni profil ile oluşturun.
            </Typography>
          )}

          {(showDraftForm || selectedId) && (
            <>
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: { xs: '1fr', md: 'minmax(260px, 2fr)' },
                  gap: 1.5,
                  alignItems: 'start',
                }}
              >
                <TextField
                  label="Profil adı"
                  value={draftName}
                  onChange={(e) => setDraftName(e.target.value)}
                  fullWidth
                  size="small"
                />
              </Box>
              {!isEvalModel && (
                <Box
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: { xs: '1fr', md: 'minmax(260px, 2fr)' },
                    gap: 1.5,
                    alignItems: 'start',
                  }}
                >
                  <TextField
                    label="Model adı"
                    value={draftModel}
                    onChange={(e) => setDraftModel(e.target.value)}
                    fullWidth
                    size="small"
                    helperText="Ollama model adı (örn. gemma3:1b)"
                  />
                </Box>
              )}
              {isEvalModel && (
                <Typography variant="body2" color="text.secondary">
                  Bu profil eval (hakem) modeli için geçerlidir.
                </Typography>
              )}
              <FormControlLabel
                control={<Switch checked={draftDefault} onChange={(_, v) => setDraftDefault(v)} />}
                label="Bu model için varsayılan profil"
              />

              <TextField
                label="System prompt (opsiyonel)"
                value={draftPrompt}
                onChange={(e) => setDraftPrompt(e.target.value)}
                fullWidth
                multiline
                minRows={8}
                placeholder="Boş bırakılabilir."
              />

              <Divider />
              <Typography variant="subtitle2">
                {isEvalModel ? 'Üretim parametreleri (opsiyonel)' : 'Ollama / üretim parametreleri (opsiyonel)'}
              </Typography>
              <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap' }}>
                {!isEvalModel && (
                  <TextField
                    size="small"
                    label="num_ctx"
                    value={draftParams.num_ctx}
                    onChange={(e) => setDraftParams((x) => ({ ...x, num_ctx: e.target.value }))}
                    sx={{ width: 120 }}
                  />
                )}
                <TextField
                  size="small"
                  label="temperature"
                  value={draftParams.temperature}
                  onChange={(e) => setDraftParams((x) => ({ ...x, temperature: e.target.value }))}
                  sx={{ width: 120 }}
                />
                <TextField
                  size="small"
                  label="top_p"
                  value={draftParams.top_p}
                  onChange={(e) => setDraftParams((x) => ({ ...x, top_p: e.target.value }))}
                  sx={{ width: 100 }}
                />
                {!isEvalModel && (
                  <>
                    <TextField
                      size="small"
                      label="repeat_penalty"
                      value={draftParams.repeat_penalty}
                      onChange={(e) => setDraftParams((x) => ({ ...x, repeat_penalty: e.target.value }))}
                      sx={{ width: 140 }}
                    />
                    <TextField
                      size="small"
                      label="num_predict"
                      value={draftParams.num_predict}
                      onChange={(e) => setDraftParams((x) => ({ ...x, num_predict: e.target.value }))}
                      sx={{ width: 120 }}
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={draftParams.think}
                          onChange={(_, v) => setDraftParams((x) => ({ ...x, think: v }))}
                        />
                      }
                      label="think (Ollama)"
                    />
                  </>
                )}
              </Stack>

              <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap' }}>
                {!selectedId ? (
                  <Button
                    variant="contained"
                    disabled={!canPersistProfile || createMut.isPending}
                    onClick={() => createMut.mutate()}
                  >
                    Kaydet (yeni)
                  </Button>
                ) : (
                  <>
                    <Button
                      variant="contained"
                      disabled={!canPersistProfile || saveMut.isPending}
                      onClick={() => saveMut.mutate()}
                    >
                      Güncelle
                    </Button>
                    <Button
                      variant="outlined"
                      disabled={cloneMut.isPending}
                      onClick={() => cloneMut.mutate(selectedId)}
                    >
                      Kopyala
                    </Button>
                    <Button
                      color="error"
                      variant="outlined"
                      disabled={deleteMut.isPending}
                      onClick={() => {
                        if (!selectedId) return
                        if (!window.confirm('Profil silinsin mi?')) return
                        deleteMut.mutate(selectedId)
                      }}
                    >
                      Sil
                    </Button>
                  </>
                )}
              </Stack>
              {selected && (
                <Typography variant="caption" color="text.secondary">
                  v{selected.version} · {selected.updated_at}
                </Typography>
              )}
            </>
          )}
        </Stack>
      </Box>
    </Stack>
  )
}
