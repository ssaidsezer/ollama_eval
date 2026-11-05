import { useMutation, useQuery } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Checkbox from '@mui/material/Checkbox'
import Chip from '@mui/material/Chip'
import FormControlLabel from '@mui/material/FormControlLabel'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { DataGrid } from '@mui/x-data-grid'
import type { GridColDef } from '@mui/x-data-grid'
import { useMemo, useState } from 'react'
import PlotLib from 'react-plotly.js'
import { api } from '../api/client'
import { useJobPolling } from '../hooks/useJobPolling'

const Plot = (PlotLib as unknown as { default?: typeof PlotLib }).default ?? PlotLib

type WhisperEntry = {
  model: string
  load_time_s: string
  avg_inference_time_s: string
  vram_allocated_gb: string
  transcriptions: Array<{
    id: string
    path: string
    original: string
    transcribed: string
    time: number
  }>
}

type QwenEntry = {
  model: string
  load_time_s: string
  avg_inference_time_s: string
  vram_allocated_gb: string
  qa_results: Array<{
    id: string
    question: string
    expected_answer: string
    generated_answer: string
    inference_time: number
    tokens_generated: number
  }>
}

type AsrReports = {
  whisper_benchmark: WhisperEntry[] | null
  qwen_benchmark: QwenEntry[] | null
  benchmark_results: unknown
  all_qa_results: unknown
  document_analysis: unknown
  all_transcriptions: unknown
  benchmark_report_text: string | null
  exists: boolean
}

const ALL_STEPS = [
  { id: 1, label: '1 · TTS Soru Üretimi' },
  { id: 2, label: '2 · Whisper Benchmark' },
  { id: 3, label: '3 · Tam Transkripsiyon' },
  { id: 4, label: '4 · Qwen Benchmark' },
  { id: 5, label: '5 · Q&A Üretimi' },
  { id: 6, label: '6 · Doküman Analizi' },
  { id: 7, label: '7 · Final Rapor' },
] as const

const parseSeconds = (s: string | undefined): number => {
  if (!s) return 0
  const m = String(s).match(/-?\d+(\.\d+)?/)
  return m ? parseFloat(m[0]) : 0
}

const parseGB = (s: string | undefined): number => parseSeconds(s)

const tokenize = (s: string): string[] =>
  s.toLocaleLowerCase('tr').replace(/[^\p{L}\p{N}\s']/gu, ' ').split(/\s+/).filter(Boolean)

const computeWer = (ref: string, hyp: string): number => {
  const r = tokenize(ref)
  const h = tokenize(hyp)
  if (!r.length) return h.length ? 1 : 0
  return wordLevenshtein(r, h) / r.length
}

function wordLevenshtein(a: string[], b: string[]): number {
  const m = a.length
  const n = b.length
  if (!m) return n
  if (!n) return m
  const dp = new Array(n + 1).fill(0).map((_, i) => i)
  for (let i = 1; i <= m; i++) {
    let prev = dp[0]
    dp[0] = i
    for (let j = 1; j <= n; j++) {
      const cur = dp[j]
      dp[j] = a[i - 1] === b[j - 1]
        ? prev
        : 1 + Math.min(prev, dp[j], dp[j - 1])
      prev = cur
    }
  }
  return dp[n]
}

export function AsrBenchmarkPage() {
  const [selectedSteps, setSelectedSteps] = useState<number[]>([1, 2, 3, 4, 5, 6, 7])
  const [skipBenchmark, setSkipBenchmark] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [whisperFilter, setWhisperFilter] = useState<string>('(Tümü)')
  const [qwenFilter, setQwenFilter] = useState<string>('(Tümü)')

  const reportsQ = useQuery<AsrReports>({
    queryKey: ['asr-reports'],
    queryFn: async () => (await api.get('/api/asr/reports')).data,
    refetchInterval: jobId ? 4000 : false,
  })

  const jobQ = useJobPolling(jobId)
  const jobStatus = jobQ.data?.status as string | undefined
  const isRunning = !!jobId && jobStatus !== 'completed' && jobStatus !== 'failed'

  const startMut = useMutation({
    mutationFn: async () => {
      const { data } = await api.post('/api/asr/run', {
        steps: selectedSteps.length === ALL_STEPS.length ? null : selectedSteps,
        skip_benchmark: skipBenchmark,
      })
      return data as { job_id: string }
    },
    onSuccess: (d) => setJobId(d.job_id),
  })

  const reports = reportsQ.data
  const whisper = reports?.whisper_benchmark ?? []
  const qwen = reports?.qwen_benchmark ?? []

  const whisperModels = useMemo(
    () => ['(Tümü)', ...whisper.map((w) => w.model)],
    [whisper],
  )
  const qwenModels = useMemo(
    () => ['(Tümü)', ...qwen.map((q) => q.model)],
    [qwen],
  )

  const whisperPerfChart = useMemo(() => {
    if (!whisper.length) return null
    const models = whisper.map((w) => w.model)
    const loads = whisper.map((w) => parseSeconds(w.load_time_s))
    const infers = whisper.map((w) => parseSeconds(w.avg_inference_time_s))
    return {
      data: [
        {
          type: 'bar' as const,
          name: 'Yükleme (s)',
          x: models,
          y: loads,
          marker: { color: '#3b82f6' },
          text: loads.map((v) => v.toFixed(2)),
          textposition: 'outside' as const,
        },
        {
          type: 'bar' as const,
          name: 'Ort. Çıkarım (s)',
          x: models,
          y: infers,
          marker: { color: '#f59e0b' },
          text: infers.map((v) => v.toFixed(2)),
          textposition: 'outside' as const,
        },
      ],
      layout: {
        title: { text: 'Whisper Yükleme ve Çıkarım Süresi' },
        barmode: 'group' as const,
        xaxis: { title: { text: 'Model' } },
        yaxis: { title: { text: 'Süre (saniye)' }, rangemode: 'tozero' as const },
        legend: { orientation: 'h' as const, y: -0.2 },
        margin: { t: 60, b: 80 },
      },
    }
  }, [whisper])

  const whisperVramChart = useMemo(() => {
    if (!whisper.length) return null
    const vrams = whisper.map((w) => parseGB(w.vram_allocated_gb))
    if (!vrams.some((v) => v > 0)) return null
    return {
      data: [
        {
          type: 'bar' as const,
          x: whisper.map((w) => w.model),
          y: vrams,
          marker: { color: '#10b981' },
          text: vrams.map((v) => `${v.toFixed(2)} GB`),
          textposition: 'outside' as const,
        },
      ],
      layout: {
        title: { text: 'Whisper VRAM Kullanımı' },
        xaxis: { title: { text: 'Model' } },
        yaxis: { title: { text: 'VRAM (GB)' }, rangemode: 'tozero' as const },
        margin: { t: 60, b: 60 },
      },
    }
  }, [whisper])

  const whisperWerChart = useMemo(() => {
    if (!whisper.length) return null
    const wers = whisper.map((w) => {
      const items = w.transcriptions ?? []
      if (!items.length) return 0
      const sum = items.reduce(
        (acc, t) => acc + computeWer(t.original ?? '', t.transcribed ?? ''),
        0,
      )
      return (sum / items.length) * 100
    })
    return {
      data: [
        {
          type: 'bar' as const,
          x: whisper.map((w) => w.model),
          y: wers,
          marker: { color: '#ef4444' },
          text: wers.map((v) => `${v.toFixed(1)}%`),
          textposition: 'auto' as const,
        },
      ],
      layout: {
        title: { text: 'Whisper Ortalama WER (%) — Daha Düşük Daha İyi' },
        xaxis: { title: { text: 'Model' } },
        yaxis: { title: { text: 'WER (%)' }, range: [0, 100] },
        margin: { t: 60, b: 60 },
      },
    }
  }, [whisper])

  const qwenChart = useMemo(() => {
    if (!qwen.length) return null
    const models = qwen.map((q) => q.model)
    const loads = qwen.map((q) => parseSeconds(q.load_time_s))
    const infers = qwen.map((q) => parseSeconds(q.avg_inference_time_s))
    return {
      data: [
        {
          type: 'bar' as const,
          name: 'Yükleme (s)',
          x: models,
          y: loads,
          marker: { color: '#3b82f6' },
          text: loads.map((v) => v.toFixed(2)),
          textposition: 'outside' as const,
        },
        {
          type: 'bar' as const,
          name: 'Ort. Çıkarım (s)',
          x: models,
          y: infers,
          marker: { color: '#f59e0b' },
          text: infers.map((v) => v.toFixed(2)),
          textposition: 'outside' as const,
        },
      ],
      layout: {
        title: { text: 'Qwen LLM Performansı' },
        barmode: 'group' as const,
        xaxis: { title: { text: 'Model' } },
        yaxis: { title: { text: 'Süre (saniye)' }, rangemode: 'tozero' as const },
        legend: { orientation: 'h' as const, y: -0.2 },
        margin: { t: 60, b: 80 },
      },
    }
  }, [qwen])

  const whisperRows = useMemo(() => {
    const rows: Array<Record<string, unknown>> = []
    let id = 0
    whisper.forEach((w) => {
      if (whisperFilter !== '(Tümü)' && w.model !== whisperFilter) return
      ;(w.transcriptions ?? []).forEach((t) => {
        const wer = computeWer(t.original ?? '', t.transcribed ?? '')
        rows.push({
          id: id++,
          model: w.model,
          q_id: t.id,
          original: t.original,
          transcribed: t.transcribed,
          wer_pct: +(wer * 100).toFixed(1),
          time_s: +Number(t.time ?? 0).toFixed(2),
          path: t.path,
        })
      })
    })
    return rows
  }, [whisper, whisperFilter])

  const whisperCols: GridColDef[] = [
    { field: 'model', headerName: 'Model', width: 160 },
    { field: 'q_id', headerName: 'Soru', width: 80 },
    { field: 'original', headerName: 'Orijinal Metin', flex: 1, minWidth: 240 },
    { field: 'transcribed', headerName: 'Transkripsiyon', flex: 1, minWidth: 240 },
    { field: 'wer_pct', headerName: 'WER (%)', width: 100, type: 'number' },
    { field: 'time_s', headerName: 'Süre (s)', width: 100, type: 'number' },
    {
      field: 'path',
      headerName: 'Ses',
      width: 280,
      renderCell: (p) => (
        <audio
          controls
          preload="none"
          src={`/api/asr/audio?path=${encodeURIComponent(String(p.value))}`}
          style={{ height: 32, width: '100%' }}
        />
      ),
    },
  ]

  const qwenRows = useMemo(() => {
    const rows: Array<Record<string, unknown>> = []
    let id = 0
    qwen.forEach((q) => {
      if (qwenFilter !== '(Tümü)' && q.model !== qwenFilter) return
      ;(q.qa_results ?? []).forEach((r) => {
        rows.push({
          id: id++,
          model: q.model,
          q_id: r.id,
          question: r.question,
          expected: r.expected_answer,
          generated: r.generated_answer,
          inference_s: +Number(r.inference_time ?? 0).toFixed(2),
          tokens: r.tokens_generated,
        })
      })
    })
    return rows
  }, [qwen, qwenFilter])

  const qwenCols: GridColDef[] = [
    { field: 'model', headerName: 'Model', width: 200 },
    { field: 'q_id', headerName: 'Soru', width: 80 },
    { field: 'question', headerName: 'Soru Metni', flex: 1, minWidth: 220 },
    { field: 'expected', headerName: 'Beklenen Cevap', flex: 1, minWidth: 220 },
    { field: 'generated', headerName: 'Üretilen Cevap', flex: 1.4, minWidth: 260 },
    { field: 'inference_s', headerName: 'Süre (s)', width: 100, type: 'number' },
    { field: 'tokens', headerName: 'Token', width: 90, type: 'number' },
  ]

  const toggleStep = (id: number) => {
    setSelectedSteps((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id].sort(),
    )
  }

  return (
    <Box>
      <Typography variant="h6">ASR Benchmark</Typography>
      <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
        ATBK ASR-LLM pipeline çıktıları: Whisper transkripsiyon, Qwen Q&amp;A ve performans karşılaştırmaları.
      </Typography>

      <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
        <Typography variant="subtitle1" sx={{ mb: 1 }}>
          Pipeline Çalıştır
        </Typography>
        <Box sx={{ mb: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {ALL_STEPS.map((s) => (
            <Chip
              key={s.id}
              label={s.label}
              color={selectedSteps.includes(s.id) ? 'primary' : 'default'}
              variant={selectedSteps.includes(s.id) ? 'filled' : 'outlined'}
              onClick={() => toggleStep(s.id)}
              disabled={isRunning}
            />
          ))}
        </Box>
        <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={skipBenchmark}
                onChange={(e) => setSkipBenchmark(e.target.checked)}
                disabled={isRunning}
              />
            }
            label="Benchmark'ları atla (--skip-benchmark)"
          />
          <Button
            variant="contained"
            disabled={isRunning || selectedSteps.length === 0}
            onClick={() => startMut.mutate()}
          >
            {isRunning ? 'Çalışıyor…' : 'Benchmark Çalıştır'}
          </Button>
          <Button
            variant="outlined"
            onClick={() => reportsQ.refetch()}
            disabled={reportsQ.isFetching}
          >
            Raporları Yenile
          </Button>
        </Box>

        {startMut.isError && (
          <Alert severity="error" sx={{ mt: 1 }}>
            Başlatma hatası: {String((startMut.error as Error)?.message ?? startMut.error)}
          </Alert>
        )}

        {jobId && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption">
              Durum: <b>{jobStatus ?? 'pending'}</b>
              {jobQ.data?.progress?.elapsed_sec
                ? ` · ${jobQ.data.progress.elapsed_sec.toFixed(0)}s`
                : ''}
            </Typography>
            {isRunning && <LinearProgress sx={{ mt: 0.5 }} />}
            {jobQ.data?.progress?.phase && (
              <Typography
                variant="caption"
                sx={{ display: 'block', fontFamily: 'monospace', mt: 0.5, opacity: 0.8 }}
              >
                {jobQ.data.progress.phase}
              </Typography>
            )}
            {jobStatus === 'failed' && (
              <Alert severity="error" sx={{ mt: 1 }}>
                Hata: {jobQ.data?.error ?? 'bilinmeyen'}
              </Alert>
            )}
            {jobStatus === 'completed' && (
              <Alert severity="success" sx={{ mt: 1 }}>
                Pipeline tamamlandı. Raporlar yenilendi.
              </Alert>
            )}
          </Box>
        )}
      </Paper>

      {reports && !reports.exists && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Henüz <code>asr/ATBK_Pipeline</code> dizini yok. Önce pipeline'ı çalıştırın.
        </Alert>
      )}

      {whisperPerfChart && (
        <Box sx={{ mt: 3, width: '100%' }}>
          <Plot
            data={whisperPerfChart.data as never}
            layout={{ ...whisperPerfChart.layout, autosize: true, height: 380 }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ displayModeBar: true, responsive: true }}
          />
        </Box>
      )}

      {whisperVramChart && (
        <Box sx={{ mt: 2, width: '100%' }}>
          <Plot
            data={whisperVramChart.data as never}
            layout={{ ...whisperVramChart.layout, autosize: true, height: 340 }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ displayModeBar: true, responsive: true }}
          />
        </Box>
      )}

      {whisperWerChart && (
        <Box sx={{ mt: 2, width: '100%' }}>
          <Plot
            data={whisperWerChart.data as never}
            layout={{ ...whisperWerChart.layout, autosize: true, height: 360 }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ displayModeBar: true, responsive: true }}
          />
        </Box>
      )}

      {whisper.length > 0 && (
        <>
          <Typography variant="h6" sx={{ mt: 3 }}>
            Whisper Transkripsiyonları
          </Typography>
          <TextField
            select
            size="small"
            label="Model"
            value={whisperFilter}
            onChange={(e) => setWhisperFilter(e.target.value)}
            sx={{ mb: 1, minWidth: 220 }}
          >
            {whisperModels.map((m) => (
              <MenuItem key={m} value={m}>
                {m}
              </MenuItem>
            ))}
          </TextField>
          <Box sx={{ height: 520, width: '100%' }}>
            <DataGrid
              rows={whisperRows}
              columns={whisperCols}
              getRowHeight={() => 64}
              density="comfortable"
            />
          </Box>
        </>
      )}

      {qwenChart && (
        <Box sx={{ mt: 3, width: '100%' }}>
          <Plot
            data={qwenChart.data as never}
            layout={{ ...qwenChart.layout, autosize: true, height: 380 }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ displayModeBar: true, responsive: true }}
          />
        </Box>
      )}

      {qwen.length > 0 && (
        <>
          <Typography variant="h6" sx={{ mt: 3 }}>
            Qwen Soru-Cevap Sonuçları
          </Typography>
          <TextField
            select
            size="small"
            label="Model"
            value={qwenFilter}
            onChange={(e) => setQwenFilter(e.target.value)}
            sx={{ mb: 1, minWidth: 220 }}
          >
            {qwenModels.map((m) => (
              <MenuItem key={m} value={m}>
                {m}
              </MenuItem>
            ))}
          </TextField>
          <Box sx={{ height: 520, width: '100%' }}>
            <DataGrid
              rows={qwenRows}
              columns={qwenCols}
              getRowHeight={() => 80}
              density="comfortable"
            />
          </Box>
        </>
      )}

      {reports?.benchmark_report_text && (
        <>
          <Typography variant="h6" sx={{ mt: 3 }}>
            Final Rapor
          </Typography>
          <Paper
            variant="outlined"
            sx={{
              p: 1.5,
              mt: 1,
              maxHeight: 480,
              overflow: 'auto',
              fontFamily: 'monospace',
              whiteSpace: 'pre',
              fontSize: 12,
            }}
          >
            {reports.benchmark_report_text}
          </Paper>
        </>
      )}
    </Box>
  )
}
