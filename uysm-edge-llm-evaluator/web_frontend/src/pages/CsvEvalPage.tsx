import { useMutation } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import LinearProgress from '@mui/material/LinearProgress'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { DataGrid } from '@mui/x-data-grid'
import type { GridColDef } from '@mui/x-data-grid'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { CollectionPicker } from '../components/CollectionPicker'
import { EvalBackendSection } from '../components/eval/EvalBackendSection'
import { EvalRagSettingsSection } from '../components/eval/EvalRagSettingsSection'
import { EvalSavedProfilesSection } from '../components/eval/EvalSavedProfilesSection'
import { QaModelPicker } from '../components/QaModelPicker'
import { SYSTEM_EVAL_PROFILE_ID } from '../eval/profileConstants'
import { useEvalPageBasics } from '../hooks/useEvalPageBasics'
import { useJobPolling } from '../hooks/useJobPolling'
import type { RagTypeUi } from '../utils/collections'

const sectionSx = { mt: 4 } as const

export function CsvEvalPage() {
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

  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [qCol, setQCol] = useState('question')
  const [aCol, setACol] = useState('answer')
  const [jobId, setJobId] = useState<string | null>(null)
  const jobQ = useJobPolling(jobId)

  const ollamaErr = ollamaQ.data?.error
  const localEvalOptions = allModels.length ? allModels : ['—']

  const startMut = useMutation({
    mutationFn: async () => {
      const fd = new FormData()
      if (csvFile) fd.append('csv_file', csvFile)
      fd.append('eval_enabled', String(evalEnabled))
      fd.append('eval_backend', evalBackend)
      fd.append('eval_model_name', evalModelName)
      if (localEvalModel) fd.append('local_eval_model_name', localEvalModel)
      fd.append('csv_question_col', qCol)
      fd.append('csv_answer_col', aCol)
      fd.append('csv_embed_model', embedModel)
      fd.append('csv_collection_name', collectionName)
      fd.append('csv_smart_rag', String(smartRag))
      fd.append('csv_retrieval_mode', retrievalMode)
      fd.append('csv_score_threshold', String(scoreTh))
      fd.append('rag_mode', ragModeApi)
      fd.append('k', String(k))
      fd.append('qa_models_json', JSON.stringify(qaSelected))
      if (openaiKey) fd.append('openai_api_key', openaiKey)
      fd.append('use_saved_qa_defaults', String(useSavedQaDefaults))
      fd.append(
        'qa_profile_by_model_json',
        JSON.stringify(useSavedQaDefaults ? {} : qaProfileByModel),
      )
      fd.append('qa_param_overrides_json', JSON.stringify({}))
      fd.append('use_saved_eval_defaults', String(useSavedEvalDefaults))
      if (evalProfileId && evalProfileId !== SYSTEM_EVAL_PROFILE_ID) {
        fd.append('eval_profile_id', evalProfileId)
      }
      fd.append('eval_param_overrides_json', JSON.stringify({}))
      const { data } = await api.post('/api/jobs/csv-pipeline', fd)
      return data.job_id as string
    },
    onSuccess: (id) => setJobId(id),
  })

  const result = jobQ.data?.result as
    | { rows: Record<string, unknown>[]; errors: string[]; csv_text: string }
    | undefined

  const [csvBlobUrl, setCsvBlobUrl] = useState<string | null>(null)
  const autoDownloadedJobs = useRef<Set<string>>(new Set())
  useEffect(() => {
    if (!result?.csv_text) {
      setCsvBlobUrl(null)
      return
    }
    const url = URL.createObjectURL(
      new Blob([result.csv_text], { type: 'text/csv;charset=utf-8' }),
    )
    setCsvBlobUrl(url)

    if (jobId && !autoDownloadedJobs.current.has(jobId)) {
      autoDownloadedJobs.current.add(jobId)
      const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
      const a = document.createElement('a')
      a.href = url
      a.download = `output_${ts}.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    }

    return () => URL.revokeObjectURL(url)
  }, [result?.csv_text, jobId])

  const cols: GridColDef[] = useMemo(() => {
    if (!result?.rows?.length) return []
    const keys = Object.keys(result.rows[0])
    return keys.map((field) => ({ field, headerName: field, flex: 1, minWidth: 120 }))
  }, [result])

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        CSV&apos;den soruları değerlendir
      </Typography>

      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Değerlendirilecek QA modelleri
        </Typography>
        {ollamaErr ? <Alert severity="error">{ollamaErr}</Alert> : null}
        <QaModelPicker
          allModels={allModels}
          filteredEmbeddingCount={ollamaQ.data?.filtered_embedding_count ?? 0}
          selected={qaSelected}
          onChange={setQaSelected}
          customModels={customModels}
          onCustomModelsChange={setCustomModels}
        />
      </Box>

      <Box sx={sectionSx}>
        <EvalSavedProfilesSection
          layout="section"
          stackMaxWidth={720}
          qaProfiles={qaProfiles}
          qaSelected={qaSelected}
          qaProfileByModel={qaProfileByModel}
          onQaProfileByModel={setQaProfileForModel}
          useSavedQaDefaults={useSavedQaDefaults}
          onUseSavedQaDefaults={setUseSavedQaDefaults}
        />
      </Box>

      <Box sx={sectionSx}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Değerlendirme
        </Typography>
        <EvalBackendSection
          variant="inlineRow"
          motorLabel="Değerlendirme motoru"
          localEvalLabel="Yerel eval modeli"
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
          localModelOptions={localEvalOptions}
          openaiKeyFieldSx={{ minWidth: 280 }}
        />
      </Box>

      <Box sx={sectionSx}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          CSV dosyası ve sütunlar
        </Typography>
        <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap', alignItems: 'center' }}>
          <Button variant="outlined" component="label">
            CSV yükle
            <input
              type="file"
              hidden
              accept=".csv"
              onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)}
            />
          </Button>
          {csvFile && (
            <Typography variant="body2" color="text.secondary">
              {csvFile.name}
            </Typography>
          )}
          <TextField
            size="small"
            label="Soru sütunu"
            value={qCol}
            onChange={(e) => setQCol(e.target.value)}
            sx={{ minWidth: 160 }}
          />
          <TextField
            size="small"
            label="Cevap sütunu"
            value={aCol}
            onChange={(e) => setACol(e.target.value)}
            sx={{ minWidth: 160 }}
          />
        </Stack>
      </Box>

      <Box
        sx={{
          ...sectionSx,
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) minmax(0, 1fr)' },
          gap: 3,
        }}
      >
        <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Embedding ve koleksiyon
          </Typography>
          <CollectionPicker
            embedModels={embedModels}
            embedLabel="Embedding modeli (indexleme ile aynı olmalı)"
            ragTypeLabel="RAG modu (indeksleme ile aynı olmalı)"
            ragHelp="BM25 anahtar kelime tabanlıdır. Smart mod parent/child koleksiyonlarını kullanır."
            horizontal
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
          title="RAG ayarları"
          ragModeUi={ragModeUi}
          onRagModeUi={setRagModeUi}
          ragModeApi={ragModeApi}
          k={k}
          onK={setK}
          scoreTh={scoreTh}
          onScoreTh={setScoreTh}
          retrievalMode={retrievalMode}
          radioRow
          kFieldLabel="k (chunk)"
          kFieldLayout="grid"
          scoreCaption={`Minimum eşleşme skoru: ${scoreTh}`}
        />
      </Box>

      <Box sx={sectionSx}>
        <Button
          variant="contained"
          disabled={startMut.isPending}
          onClick={() => startMut.mutate()}
        >
          Pipeline&apos;ı çalıştır
        </Button>
      </Box>

      {jobId && jobQ.data?.status === 'running' && <LinearProgress sx={{ mt: 2 }} />}

      {jobQ.data?.status === 'failed' && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {jobQ.data.error}
        </Alert>
      )}

      {result?.errors?.map((e, i) => (
        <Alert key={i} severity="error" sx={{ mt: 1 }}>
          {e}
        </Alert>
      ))}

      {jobQ.data?.status === 'completed' && result && (
        <Box sx={{ mt: 2 }}>
          {!result.rows?.length ? (
            <Alert severity="warning">Hiç satır üretilmedi.</Alert>
          ) : (
            <>
              <Alert severity="success">
                Pipeline tamamlandı. Toplam {result.rows.length} satır.
              </Alert>
              <Box sx={{ height: 480, width: '100%', mt: 2 }}>
                <DataGrid rows={result.rows.map((r, id) => ({ id, ...r }))} columns={cols} />
              </Box>
              <Button
                variant="outlined"
                sx={{ mt: 1 }}
                component="a"
                href={csvBlobUrl ?? '#'}
                download="output.csv"
                disabled={!csvBlobUrl}
              >
                Sonuç CSV&apos;yi indir
              </Button>
            </>
          )}
        </Box>
      )}
    </Box>
  )
}
