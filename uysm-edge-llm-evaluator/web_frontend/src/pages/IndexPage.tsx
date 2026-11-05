import { useMutation, useQuery } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import FormControlLabel from '@mui/material/FormControlLabel'
import LinearProgress from '@mui/material/LinearProgress'
import Radio from '@mui/material/Radio'
import RadioGroup from '@mui/material/RadioGroup'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useMemo, useState } from 'react'
import { api } from '../api/client'
import {
  collectionNameFull,
  smartCollectionNameFull,
} from '../utils/collections'
import { useJobPolling } from '../hooks/useJobPolling'

export function IndexPage() {
  const cfgQ = useQuery({
    queryKey: ['config'],
    queryFn: async () => (await api.get('/api/config')).data,
  })
  const embedQ = useQuery({
    queryKey: ['embed-models'],
    queryFn: async () => (await api.get('/api/models/embeddings')).data,
  })
  const ollamaQ = useQuery({
    queryKey: ['ollama-models'],
    queryFn: async () => (await api.get('/api/models/ollama')).data,
  })

  const [files, setFiles] = useState<File[]>([])
  const [indexMode, setIndexMode] = useState<'classic' | 'smart'>('classic')
  const embedModels = useMemo(() => embedQ.data?.models ?? [], [embedQ.data?.models])
  const [embedModel, setEmbedModel] = useState('')
  const [chunkSize, setChunkSize] = useState(cfgQ.data?.chunk_size_default ?? 1000)
  const [chunkOverlap, setChunkOverlap] = useState(cfgQ.data?.chunk_overlap_default ?? 200)
  const [smartParent, setSmartParent] = useState(cfgQ.data?.smart_parent_size ?? 3000)
  const [smartChild, setSmartChild] = useState(cfgQ.data?.smart_child_size ?? 500)
  const [smartOverlap, setSmartOverlap] = useState(cfgQ.data?.smart_child_overlap ?? 100)
  const [boundaryModel, setBoundaryModel] = useState(
    () => cfgQ.data?.smart_boundary_llm_model ?? 'qwen3:1.7b',
  )
  const [useRemoteIndexing, setUseRemoteIndexing] = useState(false)
  const [indexOllamaBaseUrl, setIndexOllamaBaseUrl] = useState(
    () => cfgQ.data?.indexing_ollama_base_url ?? 'http://192.168.1.151:11434',
  )

  const [jobId, setJobId] = useState<string | null>(null)
  const jobQ = useJobPolling(jobId)

  const qaModels = useMemo(() => ollamaQ.data?.models ?? [], [ollamaQ.data?.models])

  const baseName = cfgQ.data?.default_collection_name ?? 'uysm'
  const qdrantUrl = cfgQ.data?.qdrant_url ?? ''
  const selectedEmbedModel = embedModel || embedModels[0] || ''
  const configuredBoundaryModel = cfgQ.data?.smart_boundary_llm_model ?? 'qwen3:1.7b'
  const selectedBoundaryModel =
    boundaryModel || (qaModels.includes(configuredBoundaryModel) ? configuredBoundaryModel : qaModels[0]) || configuredBoundaryModel

  const preview = useMemo(() => {
    if (!selectedEmbedModel) return { text: '', smart: false }
    if (indexMode === 'smart') {
      const name = smartCollectionNameFull(
        baseName,
        selectedEmbedModel,
        smartParent,
        smartChild,
        smartOverlap,
      )
      return {
        smart: true,
        text: `Smart RAG | Embedding: ${selectedEmbedModel} | Sınır LLM: ${selectedBoundaryModel} | İndeksleme Ollama: ${useRemoteIndexing ? indexOllamaBaseUrl : 'mevcut ayar'} | Koleksiyonlar: ${name}_parents / ${name}_children | Qdrant: ${qdrantUrl}`,
      }
    }
    const name = collectionNameFull(baseName, selectedEmbedModel, chunkSize, chunkOverlap)
    return {
      smart: false,
      text: `Klasik RAG | Embedding: ${selectedEmbedModel} | İndeksleme Ollama: ${useRemoteIndexing ? indexOllamaBaseUrl : 'mevcut ayar'} | Koleksiyon: ${name} | Qdrant: ${qdrantUrl}`,
    }
  }, [
    baseName,
    indexOllamaBaseUrl,
    selectedEmbedModel,
    indexMode,
    qdrantUrl,
    smartParent,
    smartChild,
    smartOverlap,
    selectedBoundaryModel,
    chunkSize,
    chunkOverlap,
    useRemoteIndexing,
  ])

  const startMut = useMutation({
    mutationFn: async () => {
      const fd = new FormData()
      files.forEach((f) => fd.append('files', f))
      fd.append('use_smart', String(indexMode === 'smart'))
      fd.append('embed_model', selectedEmbedModel)
      fd.append('chunk_size', String(chunkSize))
      fd.append('chunk_overlap', String(chunkOverlap))
      fd.append('smart_parent_size', String(smartParent))
      fd.append('smart_child_size', String(smartChild))
      fd.append('smart_child_overlap', String(smartOverlap))
      fd.append('boundary_llm_model', selectedBoundaryModel)
      fd.append('index_ollama_base_url', useRemoteIndexing ? indexOllamaBaseUrl.trim() : '')
      const { data } = await api.post('/api/jobs/index', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      return data.job_id as string
    },
    onSuccess: (id) => setJobId(id),
  })

  const result = jobQ.data?.result as Record<string, unknown> | undefined
  const prog = jobQ.data?.progress

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        PDF&apos;leri Qdrant&apos;a indeksle (Ollama Embedding)
      </Typography>
      <Box
        sx={{
          mb: 2,
          p: 2,
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 2,
          backgroundColor: 'background.paper',
        }}
      >
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={1}
          sx={{ alignItems: { xs: 'stretch', sm: 'center' } }}
        >
          <Button variant="outlined" component="label">
            PDF yükle
            <input
              type="file"
              hidden
              multiple
              accept=".pdf"
              onChange={(e) => {
                const selected = e.target.files ? Array.from(e.target.files) : []
                if (!selected.length) return
                setFiles((prev) => {
                  const merged = [...prev]
                  selected.forEach((file) => {
                    const exists = merged.some(
                      (f) =>
                        f.name === file.name &&
                        f.size === file.size &&
                        f.lastModified === file.lastModified,
                    )
                    if (!exists) merged.push(file)
                  })
                  return merged
                })
                e.currentTarget.value = ''
              }}
            />
          </Button>
          {files.length > 0 && (
            <Button
              color="error"
              variant="text"
              onClick={() => setFiles([])}
              sx={{ alignSelf: { xs: 'flex-start', sm: 'center' } }}
            >
              Tüm PDF&apos;leri temizle
            </Button>
          )}
        </Stack>

        {files.length > 0 && (
          <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: 'wrap' }}>
            {files.map((f, idx) => (
              <Chip
                key={`${f.name}-${f.size}-${f.lastModified}-${idx}`}
                label={f.name}
                onDelete={() =>
                  setFiles((prev) => prev.filter((_, fileIndex) => fileIndex !== idx))
                }
                variant="outlined"
              />
            ))}
          </Stack>
        )}
      </Box>

      <Box
        sx={{
          mb: 2,
          p: 2,
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 2,
          backgroundColor: 'background.paper',
        }}
      >
        <Typography variant="subtitle1" sx={{ mb: 1 }}>
          Gruplama (Chunking) ayarları
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          İhtiyacınıza göre sabit chunking veya semantik smart chunking seçin.
        </Typography>

        <RadioGroup
          row
          value={indexMode}
          onChange={(_, v) => setIndexMode(v as 'classic' | 'smart')}
          sx={{ mb: 1 }}
        >
          <FormControlLabel
            value="classic"
            control={<Radio />}
            label="Klasik (sabit boyut chunking)"
          />
          <FormControlLabel
            value="smart"
            control={<Radio />}
            label="Smart (LLM semantik chunking)"
          />
        </RadioGroup>

        <Stack spacing={1.5} sx={{ mb: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={useRemoteIndexing}
                onChange={(e) => setUseRemoteIndexing(e.target.checked)}
              />
            }
            label="Embedding/chunking için 5090 sunucusunu kullan"
          />
          {useRemoteIndexing && (
            <TextField
              fullWidth
              label="5090 Ollama URL"
              value={indexOllamaBaseUrl}
              onChange={(e) => setIndexOllamaBaseUrl(e.target.value)}
              helperText="Sadece PDF indeksleme sırasında kullanılır; chat ve eval inference mevcut ayarla devam eder."
            />
          )}
        </Stack>

      {embedQ.isLoading && (
        <Alert severity="info" sx={{ my: 1 }}>
          Embedding modelleri yükleniyor...
        </Alert>
      )}

      {!embedQ.isLoading && embedModels.length === 0 && (
        <Alert severity="error" sx={{ my: 1 }}>
          Embedding modelleri yüklenemedi: {embedQ.data?.error || 'Bilinmeyen hata'}
        </Alert>
      )}

      {embedModels.length > 0 && (
        <>
          <TextField
            select
            fullWidth
            label="Embedding modeli"
            value={selectedEmbedModel}
            onChange={(e) => setEmbedModel(e.target.value)}
            slotProps={{ select: { native: true } }}
            sx={{ mb: 1.5 }}
          >
            {embedModels.map((m: string) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </TextField>

          {indexMode === 'smart' ? (
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mb: 2 }}>
              <TextField
                label="Parent blok boyutu (karakter)"
                type="number"
                value={smartParent}
                onChange={(e) => setSmartParent(+e.target.value)}
              />
              <TextField
                label="Child chunk boyutu"
                type="number"
                value={smartChild}
                onChange={(e) => setSmartChild(+e.target.value)}
              />
              <TextField
                label="Child overlap"
                type="number"
                value={smartOverlap}
                onChange={(e) => setSmartOverlap(+e.target.value)}
              />
              <TextField
                select
                label="Sınır LLM"
                value={selectedBoundaryModel}
                onChange={(e) => setBoundaryModel(e.target.value)}
                slotProps={{ select: { native: true } }}
                fullWidth
              >
                {(qaModels.length ? qaModels : ['Model Yok']).map((m: string) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </TextField>
            </Stack>
          ) : (
            <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
              <TextField
                label="Chunk boyutu"
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(+e.target.value)}
              />
              <TextField
                label="Chunk overlap"
                type="number"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(+e.target.value)}
              />
            </Stack>
          )}
        </>
      )}
      </Box>

      {preview.text && <Alert severity="info">{preview.text}</Alert>}
      {indexMode === 'smart' && smartChild >= smartParent && (
        <Alert severity="warning" sx={{ mt: 1 }}>
          Child boyutu parent’a eşit veya büyük — Smart chunking avantajı azalır.
        </Alert>
      )}

      <Button
        variant="contained"
        sx={{ mt: 2 }}
        disabled={
          !selectedEmbedModel ||
          files.length === 0 ||
          startMut.isPending ||
          (useRemoteIndexing && !indexOllamaBaseUrl.trim())
        }
        onClick={() => startMut.mutate()}
      >
        İndeksi oluştur / güncelle
      </Button>

      {jobId && jobQ.data?.status === 'running' && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption">
            {prog?.phase} — {prog?.current}/{prog?.total} ({prog?.elapsed_sec?.toFixed?.(1)}s)
          </Typography>
          <LinearProgress />
        </Box>
      )}

      {jobQ.data?.status === 'failed' && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {jobQ.data.error}
        </Alert>
      )}

      {jobQ.data?.status === 'completed' && result && (
        <Box sx={{ mt: 2 }}>
          {'total_chunks' in result ? (
            <>
              <Alert severity="success">
                İndeksleme tamamlandı! Toplam {String(result.total_chunks)} chunk.
              </Alert>
              <Typography variant="subtitle2" sx={{ mt: 1 }}>
                Süre: PDF {String(result.pdf_extract_sec)}s | Embed{' '}
                {String(result.ollama_embed_sec)}s | Qdrant {String(result.qdrant_upsert_sec)}
                s | Toplam {String(result.total_sec)}s
              </Typography>
            </>
          ) : (
            <>
              <Alert severity="success">
                Smart indeksleme tamamlandı! Toplam {String(result.total_parents)} parent,{' '}
                {String(result.total_children)} child.
              </Alert>
              <Typography variant="subtitle2" sx={{ mt: 1 }}>
                PDF {String(result.pdf_extract_sec)}s | LLM {String(result.llm_boundary_sec)}s |
                Parent store {String(result.parent_store_sec)}s | Child embed{' '}
                {String(result.child_embed_sec)}s | Child upsert {String(result.child_upsert_sec)}
                s | Toplam {String(result.total_sec)}s
              </Typography>
              {Number(result.timeout_count) > 0 && (
                <Alert severity="warning" sx={{ mt: 1 }}>
                  LLM boundary: {String(result.timeout_count)} timeout
                </Alert>
              )}
            </>
          )}
        </Box>
      )}
    </Box>
  )
}
