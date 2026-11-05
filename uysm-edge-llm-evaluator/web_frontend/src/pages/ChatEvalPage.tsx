import { useMutation } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Collapse from '@mui/material/Collapse'
import CircularProgress from '@mui/material/CircularProgress'
import Divider from '@mui/material/Divider'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { api } from '../api/client'
import { CollectionPicker } from '../components/CollectionPicker'
import { EvalBackendSection } from '../components/eval/EvalBackendSection'
import { EvalRagSettingsSection } from '../components/eval/EvalRagSettingsSection'
import { EvalSavedProfilesSection } from '../components/eval/EvalSavedProfilesSection'
import { QaModelPicker } from '../components/QaModelPicker'
import { SYSTEM_EVAL_PROFILE_ID } from '../eval/profileConstants'
import { useEvalPageBasics } from '../hooks/useEvalPageBasics'
import type { RagTypeUi } from '../utils/collections'

function chunkTitle(retrievalMode: string, smartRag: boolean) {
  if (retrievalMode === 'bm25' && smartRag)
    return 'Retrieved Chunks — BM25 Smart (child arama, parent bağlam)'
  if (retrievalMode === 'bm25') return 'Retrieved Chunks — BM25 Klasik'
  if (smartRag) return 'Retrieved Chunks — Smart RAG'
  return 'Retrieved Chunks'
}

export function ChatEvalPage() {
  const {
    ollamaQ,
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
    setQaProfileForModel,
    useSavedEvalDefaults,
    evalProfileId,
    setEvalProfileId,
    smartRag,
    retrievalMode,
    ragModeApi,
    qaProfiles,
    evalProfiles,
  } = useEvalPageBasics()

  const [question, setQuestion] = useState('')
  const [expected, setExpected] = useState('')
  const [formError, setFormError] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [resp, setResp] = useState<any>(null)

  const evalMut = useMutation({
    mutationFn: async () => {
      const { data } = await api.post('/api/chat/eval', {
        question: question.trim(),
        expected_answer: expected,
        rag_mode: ragModeApi,
        k,
        qa_models_selected: qaSelected,
        eval_enabled: evalEnabled,
        eval_backend: evalBackend,
        eval_model_name: evalModelName,
        local_eval_model_name: localEvalModel || null,
        openai_api_key: openaiKey || null,
        collection_name: collectionName,
        embed_model: embedModel || null,
        smart_rag: smartRag,
        score_threshold: scoreTh,
        retrieval_mode: retrievalMode,
        use_saved_qa_defaults: useSavedQaDefaults,
        qa_profile_by_model: useSavedQaDefaults ? {} : qaProfileByModel,
        use_saved_eval_defaults: useSavedEvalDefaults,
        eval_profile_id: evalProfileId === SYSTEM_EVAL_PROFILE_ID ? null : evalProfileId,
      })
      return data
    },
    onMutate: () => {
      setIsGenerating(true)
      setFormError(null)
      setResp(null)
    },
    onSuccess: (data) => setResp(data),
    onError: (err: any) => {
      const apiMessage =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        'Yanıtlar üretilirken bir hata oluştu.'
      setFormError(String(apiMessage))
    },
    onSettled: () => setIsGenerating(false),
  })

  const downloadChatCsv = () => {
    if (!resp?.chat_eval_rows?.length) return
    const rows = resp.chat_eval_rows as Record<string, unknown>[]
    const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))]
    const esc = (v: unknown) => {
      const s = v == null ? '' : String(v)
      if (s.includes('"') || s.includes(',') || s.includes('\n')) return `"${s.replace(/"/g, '""')}"`
      return s
    }
    const lines = [keys.join(','), ...rows.map((r) => keys.map((k) => esc(r[k])).join(','))]
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'chat_results.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Box>
      {ollamaQ.data?.error && <Alert severity="error">{ollamaQ.data.error}</Alert>}
      <QaModelPicker
        allModels={allModels}
        filteredEmbeddingCount={ollamaQ.data?.filtered_embedding_count ?? 0}
        selected={qaSelected}
        onChange={setQaSelected}
        customModels={customModels}
        onCustomModelsChange={setCustomModels}
      />

      <EvalSavedProfilesSection
        layout="boxed"
        qaProfiles={qaProfiles}
        qaSelected={qaSelected}
        qaProfileByModel={qaProfileByModel}
        onQaProfileByModel={setQaProfileForModel}
        useSavedQaDefaults={useSavedQaDefaults}
        onUseSavedQaDefaults={setUseSavedQaDefaults}
      />

      <Box sx={{ mb: 2 }}>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', lg: 'repeat(3, minmax(0, 1fr))' },
            gap: 2,
          }}
        >
          <EvalBackendSection
            variant="gridCard"
            evalEnabled={evalEnabled}
            onEvalEnabled={setEvalEnabled}
            evalBackend={evalBackend}
            onEvalBackend={setEvalBackend}
            evalModelName={evalModelName}
            onEvalModelName={setEvalModelName}
            localEvalModel={localEvalModel}
            onLocalEvalModel={setLocalEvalModel}
            openaiKey={openaiKey}
            onOpenaiKey={setOpenaiKey}
            evalProfiles={evalProfiles}
            evalProfileId={evalProfileId}
            onEvalProfileId={setEvalProfileId}
            localModelOptions={allModels}
          />
          <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Embedding ve retrieval
            </Typography>
            <CollectionPicker
              embedModels={embedModels}
              embedLabel="Embedding modeli"
              ragTypeLabel="Retrieval tipi"
              ragHelp="Smart parent/child; BM25 anahtar kelime."
              horizontal={false}
              classicLabel="Klasik koleksiyon"
              smartLabel="Smart koleksiyon"
              embedModel={embedModel}
              onEmbedModel={setEmbedModel}
              ragType={ragType as RagTypeUi}
              onRagType={setRagType}
              collectionName={collectionName}
              onCollectionName={setCollectionName}
            />
          </Box>
          <EvalRagSettingsSection
            ragModeUi={ragModeUi}
            onRagModeUi={setRagModeUi}
            ragModeApi={ragModeApi}
            k={k}
            onK={setK}
            scoreTh={scoreTh}
            onScoreTh={setScoreTh}
            retrievalMode={retrievalMode}
            radioRow={false}
            kFieldLayout="stack"
            scoreCaption={`Score threshold: ${scoreTh}`}
          />
        </Box>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Soru ve referans
        </Typography>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
          <TextField
            label="Soru"
            multiline
            minRows={6}
            fullWidth
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <TextField
            label="Beklenen / referans cevap"
            multiline
            minRows={6}
            fullWidth
            value={expected}
            onChange={(e) => setExpected(e.target.value)}
          />
        </Stack>
      </Box>

      <Button
        variant="contained"
        fullWidth
        disabled={isGenerating}
        onClick={() => {
          const q = question.trim()
          if (!q) {
            setFormError('Lütfen bir soru gir.')
            return
          }
          if (ragModeApi !== 'no_rag' && !collectionName.trim()) {
            setFormError('Seçili retrieval tipi için geçerli bir koleksiyon bulunamadı.')
            return
          }
          evalMut.mutate()
        }}
      >
        {isGenerating ? 'Yanıtlar üretiliyor...' : 'Soruyu değerlendir'}
      </Button>

      {isGenerating && (
        <Alert severity="info" sx={{ mt: 2 }}>
          <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
            <CircularProgress size={16} />
            <Typography variant="body2">
              Yanıtlar üretiliyor. İşlem bitince buton otomatik açılacak.
            </Typography>
          </Stack>
        </Alert>
      )}

      {formError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {formError}
        </Alert>
      )}

      {resp?.errors?.length > 0 &&
        resp.errors.map((e: string, i: number) => (
          <Alert key={i} severity="error" sx={{ mt: 2 }}>
            {e}
          </Alert>
        ))}

      {resp && (
        <Box sx={{ mt: 3 }}>
          {resp.chunk_warning && <Alert severity="info">{resp.chunk_warning}</Alert>}
          {resp.chunks?.length > 0 && (
            <>
              <Typography variant="subtitle1" sx={{ mt: 2 }}>
                {chunkTitle(retrievalMode, smartRag)}
              </Typography>
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
                  gap: 1,
                  mt: 1,
                }}
              >
                {resp.chunks.map((c: any, i: number) => (
                  <Paper
                    key={i}
                    variant="outlined"
                    sx={{ p: 1, borderColor: 'divider', bgcolor: 'background.paper' }}
                  >
                    <Typography variant="caption" color="primary">
                      Chunk {i + 1} — {smartRag ? 'Child / Parent' : ''}
                    </Typography>
                    {smartRag && c.child_text && (
                      <Typography
                        variant="caption"
                        sx={{ display: 'block', maxHeight: 60, overflow: 'auto', color: '#aaa' }}
                      >
                        {c.child_text}
                      </Typography>
                    )}
                    <Typography
                      variant="body2"
                      sx={{ maxHeight: 100, overflow: 'auto', mt: 0.5, fontSize: '0.78rem' }}
                    >
                      {c.text}
                    </Typography>
                    <Typography variant="caption" sx={{ float: 'right', color: '#4caf50' }}>
                      {Number(c.score).toFixed(3)}
                    </Typography>
                  </Paper>
                ))}
              </Box>
            </>
          )}

          <Divider sx={{ my: 2 }} />

          {resp.model_results?.map((mr: any) => (
            <Box key={mr.model_name} sx={{ mb: 3 }}>
              <Typography variant="h6">{mr.model_name}</Typography>
              {mr.profile_trace && Object.keys(mr.profile_trace).length > 0 && (
                <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                  Profil izi: {JSON.stringify(mr.profile_trace)}
                </Typography>
              )}
              {mr.error && <Alert severity="error">{mr.error}</Alert>}
              {ragModeApi === 'both' && !mr.error ? (
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                  <Paper sx={{ p: 2, flex: 1 }}>
                    <Typography sx={{ fontWeight: 'bold' }}>RAG&apos;li</Typography>
                    <Typography sx={{ whiteSpace: 'pre-wrap' }}>{mr.rag_answer}</Typography>
                    <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.8 }}>
                      Model meta
                    </Typography>
                    <pre style={{ fontSize: 12, overflow: 'auto' }}>
                      {JSON.stringify(mr.rag_result_meta ?? {}, null, 2)}
                    </pre>
                    <Collapse in={!!mr.rag_eval && evalEnabled}>
                      <pre style={{ fontSize: 12, overflow: 'auto' }}>
                        {JSON.stringify(mr.rag_eval, null, 2)}
                      </pre>
                    </Collapse>
                  </Paper>
                  <Paper sx={{ p: 2, flex: 1 }}>
                    <Typography sx={{ fontWeight: 'bold' }}>RAG&apos;siz</Typography>
                    <Typography sx={{ whiteSpace: 'pre-wrap' }}>{mr.no_rag_answer}</Typography>
                    <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.8 }}>
                      Model meta
                    </Typography>
                    <pre style={{ fontSize: 12, overflow: 'auto' }}>
                      {JSON.stringify(mr.no_rag_result_meta ?? {}, null, 2)}
                    </pre>
                    <Collapse in={!!mr.no_rag_eval && evalEnabled}>
                      <pre style={{ fontSize: 12, overflow: 'auto' }}>
                        {JSON.stringify(mr.no_rag_eval, null, 2)}
                      </pre>
                    </Collapse>
                  </Paper>
                </Stack>
              ) : (
                !mr.error && (
                  <>
                    <Typography sx={{ whiteSpace: 'pre-wrap', mt: 1 }}>
                      {ragModeApi === 'rag' ? mr.rag_answer : mr.no_rag_answer}
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.8 }}>
                      Model meta
                    </Typography>
                    <pre style={{ fontSize: 12, overflow: 'auto' }}>
                      {JSON.stringify(
                        ragModeApi === 'rag' ? mr.rag_result_meta ?? {} : mr.no_rag_result_meta ?? {},
                        null,
                        2,
                      )}
                    </pre>
                    <Collapse in={evalEnabled}>
                      <pre style={{ fontSize: 12 }}>
                        {JSON.stringify(
                          ragModeApi === 'rag' ? mr.rag_eval : mr.no_rag_eval,
                          null,
                          2,
                        )}
                      </pre>
                    </Collapse>
                  </>
                )
              )}
              <Divider sx={{ mt: 2 }} />
            </Box>
          ))}

          {resp.chat_eval_rows?.length > 0 && (
            <Button variant="outlined" onClick={downloadChatCsv} sx={{ mt: 2 }}>
              Manuel chat sonuçlarını CSV indir
            </Button>
          )}
        </Box>
      )}
    </Box>
  )
}
