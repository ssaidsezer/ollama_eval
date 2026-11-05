import { useQuery } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Stack from '@mui/material/Stack'
import { api } from '../api/client'

export function ConnectionHeader() {
  const q = useQuery({
    queryKey: ['connection-status'],
    queryFn: async () => (await api.get('/api/connection-status')).data,
    refetchInterval: 1_000,
    refetchIntervalInBackground: true,
  })
  if (!q.data) return null
  const d = q.data
  const m = d.monitor ?? {}
  const cpuUsage = Number(m.cpu_usage ?? 0)
  const gpuUsage = Number(m.gpu_usage ?? 0)
  const ramUsed = m.ram_used
  const ramTotal = m.ram_total
  const storageUsed = m.storage_used
  const storageTotal = m.storage_total

  const formatSizeFromMB = (valueMb: number) => {
    if (valueMb >= 1024 * 1024) return `${(valueMb / (1024 * 1024)).toFixed(1)} TB`
    if (valueMb >= 1024) return `${(valueMb / 1024).toFixed(1)} GB`
    return `${valueMb.toFixed(1)} MB`
  }

  const formatUsage = (used: unknown, total: unknown) => {
    if (used == null || total == null) return 'N/A'
    const usedNum = Number(used)
    const totalNum = Number(total)
    if (!Number.isFinite(usedNum) || !Number.isFinite(totalNum) || totalNum <= 0) {
      return `${used}/${total}`
    }
    const pct = (usedNum / totalNum) * 100
    return `${formatSizeFromMB(usedNum)} / ${formatSizeFromMB(totalNum)} (%${pct.toFixed(1)})`
  }

  return (
    <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap' }}>
      <Box sx={{ flex: '1 1 200px', minWidth: 180 }}>
        {d.qdrant_ok ? (
          <Alert severity="success">{d.qdrant_message}</Alert>
        ) : (
          <Alert severity="error">{d.qdrant_message}</Alert>
        )}
      </Box>
      <Box sx={{ flex: '1 1 200px', minWidth: 180 }}>
        {d.ollama_ok ? (
          <Alert severity="success">{d.ollama_message}</Alert>
        ) : (
          <Alert severity="error">{d.ollama_message}</Alert>
        )}
      </Box>
      <Box sx={{ flex: '2 1 280px', minWidth: 220 }}>
        {d.monitor ? (
          <Alert severity="success">
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 0.5,
                alignItems: 'center',
              }}
            >
              <Box>CPU: %{cpuUsage.toFixed(1)}</Box>
              <Box>RAM: {formatUsage(ramUsed, ramTotal)}</Box>
              <Box>GPU: %{gpuUsage.toFixed(1)}</Box>
              <Box>Storage: {formatUsage(storageUsed, storageTotal)}</Box>
            </Box>
          </Alert>
        ) : (
          <Alert severity="warning">
            Sunucu: {d.monitor_error || 'Monitor yok'}
          </Alert>
        )}
      </Box>
    </Stack>
  )
}
