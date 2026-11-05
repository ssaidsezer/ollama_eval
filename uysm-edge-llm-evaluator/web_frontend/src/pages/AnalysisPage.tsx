import { useMutation } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import FormControlLabel from '@mui/material/FormControlLabel'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Radio from '@mui/material/Radio'
import RadioGroup from '@mui/material/RadioGroup'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { DataGrid } from '@mui/x-data-grid'
import type { GridColDef } from '@mui/x-data-grid'
import { useMemo, useState } from 'react'
import PlotLib from 'react-plotly.js'
import { api } from '../api/client'

const Plot = (PlotLib as unknown as { default?: typeof PlotLib }).default ?? PlotLib

export function AnalysisPage() {
  const [result, setResult] = useState<any>(null)
  const [problemMode, setProblemMode] = useState<'worse' | 'low'>('worse')
  const [filterModel, setFilterModel] = useState('(Tümü)')
  const [filterVerdict, setFilterVerdict] = useState('(Tümü)')
  const [filterRag, setFilterRag] = useState('(Tümü)')
  const [selectedRow, setSelectedRow] = useState<Record<string, unknown> | null>(null)

  const mut = useMutation({
    mutationFn: async (file: File) => {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await api.post('/api/analysis', fd)
      return data
    },
    onSuccess: (d) => setResult(d),
  })

  const gs = result?.global_summary

  const breakdownCols: GridColDef[] = useMemo(() => {
    if (!result?.breakdown?.length) return []
    return Object.keys(result.breakdown[0]).map((field) => ({
      field,
      headerName: field,
      flex: 1,
      minWidth: 100,
    }))
  }, [result])

  const liftCols: GridColDef[] = useMemo(() => {
    if (!result?.lift_rows?.length) return []
    return Object.keys(result.lift_rows[0]).map((field) => ({
      field,
      headerName: field,
      flex: 1,
      minWidth: 100,
    }))
  }, [result])

  const detailRowsFiltered = useMemo(() => {
    if (!result?.detail_rows) return []
    let df = result.detail_rows as Record<string, unknown>[]
    if (filterModel !== '(Tümü)') df = df.filter((r) => r.model === filterModel)
    if (filterVerdict !== '(Tümü)') df = df.filter((r) => r.ai_verdict === filterVerdict)
    if (filterRag !== '(Tümü)') df = df.filter((r) => r.rag_type === filterRag)
    return df
  }, [result, filterModel, filterVerdict, filterRag])

  const detailCols: GridColDef[] = useMemo(() => {
    const preferred = [
      'model',
      'question',
      'rag_type',
      'ai_verdict',
      'ai_score',
      'ai_hallucination_risk',
      'tokens_per_second',
      'response_time_seconds',
      'eval_duration_seconds',
      'model_answer',
      'answer',
    ]
    if (!detailRowsFiltered.length) return []
    const keys = [...new Set([...preferred, ...Object.keys(detailRowsFiltered[0])])].filter(
      (k) => k in detailRowsFiltered[0],
    )
    return keys.map((field) => ({ field, headerName: field, flex: 1, minWidth: 120 }))
  }, [detailRowsFiltered])

  const modelOpts = useMemo(() => {
    if (!result?.detail_rows?.length) return ['(Tümü)']
    const u = [...new Set(result.detail_rows.map((r: any) => r.model).filter(Boolean))] as string[]
    return ['(Tümü)', ...u.sort()]
  }, [result])

  const verdictOpts = useMemo(() => {
    if (!result?.detail_rows?.length) return ['(Tümü)']
    const u = [...new Set(result.detail_rows.map((r: any) => r.ai_verdict).filter(Boolean))] as string[]
    return ['(Tümü)', ...u.sort()]
  }, [result])

  const ragOpts = useMemo(() => {
    if (!result?.detail_rows?.length) return ['(Tümü)']
    const u = [...new Set(result.detail_rows.map((r: any) => r.rag_type).filter(Boolean))] as string[]
    return ['(Tümü)', ...u.sort()]
  }, [result])

  const problemData =
    problemMode === 'worse' ? result?.problems_rag_worse : result?.problems_both_low

  const problemCols: GridColDef[] = useMemo(() => {
    if (!problemData?.length) return []
    return Object.keys(problemData[0]).map((field) => ({
      field,
      headerName: field,
      flex: 1,
      minWidth: 100,
    }))
  }, [problemData])

  return (
    <Box>
      <Typography variant="h6">Sonuç Analizi</Typography>
      <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
        Export CSV (noktalı virgül) yükleyin.
      </Typography>

      <Button variant="outlined" component="label">
        Export CSV yükle
        <input
          type="file"
          hidden
          accept=".csv"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) mut.mutate(f)
          }}
        />
      </Button>

      {result?.warnings?.map((w: string, i: number) => (
        <Alert key={i} severity="warning" sx={{ mt: 1 }}>
          {w}
        </Alert>
      ))}

      {result?.info_messages?.map((m: string, i: number) => (
        <Alert key={i} severity="info" sx={{ mt: 1 }}>
          {m}
        </Alert>
      ))}

      {gs && (
        <>
          <Typography variant="h6" sx={{ mt: 3 }}>
            Genel Özet
          </Typography>
          <Typography variant="caption">
            Toplam {gs.row_count} satır · RAG ort. skor {gs.rag_mean_score?.toFixed?.(2) ?? '—'} ·
            RAG&apos;siz {gs.no_rag_mean_score?.toFixed?.(2) ?? '—'} (Δ {gs.score_delta?.toFixed?.(2) ?? '—'})
          </Typography>
        </>
      )}

      {result?.breakdown?.length > 0 && (
        <>
          <Typography variant="h6" sx={{ mt: 3 }}>
            Model × RAG kırılımı
          </Typography>
          <Box sx={{ height: 360, width: '100%' }}>
            <DataGrid
              rows={result.breakdown.map((r: object, id: number) => ({ id, ...r }))}
              columns={breakdownCols}
            />
          </Box>
        </>
      )}

      {result?.charts?.map((c: { id: string; title: string; plotly: any }) => (
        <Box key={c.id} sx={{ mt: 2, width: '100%' }}>
          <Plot
            data={c.plotly.data}
            layout={{
              ...c.plotly.layout,
              autosize: true,
              height: 420,
              hoverlabel: {
                bgcolor: '#111827',
                bordercolor: '#374151',
                font: { color: '#f9fafb' },
              },
            }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ displayModeBar: true, responsive: true }}
          />
        </Box>
      ))}

      {result?.distribution_chart && (
      <Box sx={{ mt: 2 }}>
          <Plot
            data={result.distribution_chart.data}
            layout={{
              ...result.distribution_chart.layout,
              autosize: true,
              height: 420,
              hoverlabel: {
                bgcolor: '#111827',
                bordercolor: '#374151',
                font: { color: '#f9fafb' },
              },
            }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </Box>
      )}

      {result?.lift_rows?.length > 0 && (
        <>
          <Typography variant="h6" sx={{ mt: 3 }}>
            RAG Lift
          </Typography>
          <Box sx={{ height: 320, width: '100%' }}>
            <DataGrid
              rows={result.lift_rows.map((r: object, id: number) => ({ id, ...r }))}
              columns={liftCols}
            />
          </Box>
        </>
      )}

      {problemData && (
        <>
          <Typography variant="h6" sx={{ mt: 3 }}>
            Problemli sorular
          </Typography>
          <RadioGroup
            row
            value={problemMode}
            onChange={(_, v) => setProblemMode(v as 'worse' | 'low')}
          >
            <FormControlLabel
              value="worse"
              control={<Radio />}
              label="RAG'siz > RAG'li (RAG kötüleştirmiş)"
            />
            <FormControlLabel
              value="low"
              control={<Radio />}
              label="Her ikisi de ≤ 5"
            />
          </RadioGroup>
          <Typography variant="caption">{problemData.length} satır</Typography>
          {problemData.length > 0 && (
            <Box sx={{ height: 400, width: '100%' }}>
              <DataGrid
                rows={problemData.map((r: object, id: number) => ({ id, ...r }))}
                columns={problemCols}
                onRowClick={(params) => setSelectedRow(params.row as Record<string, unknown>)}
              />
            </Box>
          )}
        </>
      )}

      {result?.detail_rows?.length > 0 && (
        <>
          <Typography variant="h6" sx={{ mt: 3 }}>
            Soru bazlı detay
          </Typography>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mb: 1 }}>
            <TextField
              select
              label="Model"
              size="small"
              value={filterModel}
              onChange={(e) => setFilterModel(e.target.value)}
            >
              {modelOpts.map((o) => (
                <MenuItem key={o} value={o}>
                  {o}
                </MenuItem>
              ))}
            </TextField>
            <TextField
              select
              label="Verdict"
              size="small"
              value={filterVerdict}
              onChange={(e) => setFilterVerdict(e.target.value)}
            >
              {verdictOpts.map((o) => (
                <MenuItem key={o} value={o}>
                  {o}
                </MenuItem>
              ))}
            </TextField>
            <TextField
              select
              label="RAG Türü"
              size="small"
              value={filterRag}
              onChange={(e) => setFilterRag(e.target.value)}
            >
              {ragOpts.map((o) => (
                <MenuItem key={o} value={o}>
                  {o}
                </MenuItem>
              ))}
            </TextField>
          </Stack>
          <Typography variant="caption">
            {detailRowsFiltered.length} / {result.detail_rows.length} satır
          </Typography>
          <Box sx={{ height: 520, width: '100%' }}>
            <DataGrid
              rows={detailRowsFiltered.map((r: object, id: number) => ({ id, ...r }))}
              columns={detailCols}
              onRowClick={(params) => setSelectedRow(params.row as Record<string, unknown>)}
            />
          </Box>
        </>
      )}
      {selectedRow && (
        <Paper variant="outlined" sx={{ mt: 2, p: 1.5 }}>
          <Stack
            direction="row"
            sx={{ mb: 1, justifyContent: 'space-between', alignItems: 'center' }}
          >
            <Typography variant="subtitle2">Seçili Satır Detayı</Typography>
            <Button size="small" onClick={() => setSelectedRow(null)}>
              Kapat
            </Button>
          </Stack>
          <Stack spacing={1}>
            {Object.entries(selectedRow)
              .filter(([k]) => k !== 'id')
              .map(([key, value]) => (
                <Paper key={key} variant="outlined" sx={{ p: 1.25 }}>
                  <Typography variant="caption" color="text.secondary">
                    {key}
                  </Typography>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {value == null ? '—' : String(value)}
                  </Typography>
                </Paper>
              ))}
          </Stack>
        </Paper>
      )}
    </Box>
  )
}
