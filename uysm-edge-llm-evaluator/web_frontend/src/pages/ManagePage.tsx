import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import MenuItem from '@mui/material/MenuItem'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { api } from '../api/client'

export function ManagePage() {
  const qc = useQueryClient()
  const rawQ = useQuery({
    queryKey: ['ollama-raw'],
    queryFn: async () => (await api.get('/api/ollama/models/all-raw')).data,
  })
  const colQ = useQuery({
    queryKey: ['collections'],
    queryFn: async () => (await api.get('/api/qdrant/collections')).data,
  })

  const [newModel, setNewModel] = useState('')
  const [delModel, setDelModel] = useState('')
  const [delColl, setDelColl] = useState('')
  const [confirmModel, setConfirmModel] = useState(false)
  const [confirmColl, setConfirmColl] = useState(false)

  const allNames: string[] = rawQ.data?.models ?? []
  const collections: string[] = colQ.data?.collections ?? []

  const pullMut = useMutation({
    mutationFn: async () => (await api.post('/api/ollama/models/pull', { name: newModel.trim() })).data,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['ollama-raw'] })
      void qc.invalidateQueries({ queryKey: ['ollama-models'] })
      void qc.invalidateQueries({ queryKey: ['embed-models'] })
      setNewModel('')
    },
  })

  const delModelMut = useMutation({
    mutationFn: async () => {
      const name = delModel || allNames[0] || ''
      return (await api.delete('/api/ollama/models', { data: { name } })).data
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['ollama-raw'] })
      void qc.invalidateQueries({ queryKey: ['ollama-models'] })
      void qc.invalidateQueries({ queryKey: ['embed-models'] })
      setConfirmModel(false)
    },
  })

  const delCollMut = useMutation({
    mutationFn: async () => {
      const name = delColl || collections[0] || ''
      return (await api.delete(`/api/qdrant/collections/${encodeURIComponent(name)}`)).data
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['collections'] })
      void qc.invalidateQueries({ queryKey: ['col-opts'] })
      setConfirmColl(false)
    },
  })

  return (
    <Stack direction={{ xs: 'column', md: 'row' }} spacing={3}>
      <Box sx={{ flex: 1 }}>
        <Typography variant="h6">Ollama Model Yönetimi</Typography>
        {rawQ.data?.error && <Alert severity="error">{rawQ.data.error}</Alert>}
        <Stack direction="row" spacing={1} sx={{ mt: 2, alignItems: 'stretch' }}>
          <TextField
            fullWidth
            size="small"
            placeholder="örn: llama3.2:3b"
            value={newModel}
            onChange={(e) => setNewModel(e.target.value)}
          />
          <Button
            variant="contained"
            size="small"
            sx={{ height: 40, minWidth: 140, whiteSpace: 'nowrap' }}
            onClick={() => newModel.trim() && pullMut.mutate()}
            disabled={pullMut.isPending}
          >
            Ekle / Pull Et
          </Button>
        </Stack>
        {pullMut.data && (
          <Alert severity={pullMut.data.success ? 'success' : 'error'} sx={{ mt: 1 }}>
            {pullMut.data.message}
          </Alert>
        )}

        {allNames.length > 0 && (
          <>
            <Typography variant="caption" sx={{ mt: 2, display: 'block' }}>
              Toplam {allNames.length} model
            </Typography>
            <TextField
              select
              fullWidth
              label="Silinecek model"
              value={delModel || allNames[0] || ''}
              onChange={(e) => setDelModel(e.target.value)}
              sx={{ mt: 1 }}
            >
              {allNames.map((m) => (
                <MenuItem key={m} value={m}>
                  {m}
                </MenuItem>
              ))}
            </TextField>
            <Button color="error" sx={{ mt: 1 }} onClick={() => setConfirmModel(true)}>
              Modeli Sil
            </Button>
          </>
        )}
      </Box>

      <Box sx={{ flex: 1 }}>
        <Typography variant="h6">Qdrant Koleksiyon Yönetimi</Typography>
        {colQ.data?.error && <Alert severity="error">{colQ.data.error}</Alert>}
        {!colQ.data?.error && collections.length === 0 && (
          <Alert severity="info">Hiç koleksiyon yok.</Alert>
        )}
        {collections.length > 0 && (
          <>
            <Typography variant="caption">Toplam {collections.length} koleksiyon</Typography>
            <TextField
              select
              fullWidth
              label="Silinecek koleksiyon"
              value={delColl || collections[0]}
              onChange={(e) => setDelColl(e.target.value)}
              sx={{ mt: 1 }}
            >
              {collections.map((c) => (
                <MenuItem key={c} value={c}>
                  {c}
                </MenuItem>
              ))}
            </TextField>
            <Button color="error" sx={{ mt: 1 }} onClick={() => setConfirmColl(true)}>
              Koleksiyonu Sil
            </Button>
          </>
        )}
      </Box>

      <Dialog open={confirmModel} onClose={() => setConfirmModel(false)}>
        <DialogTitle>Model silinsin mi?</DialogTitle>
        <DialogContent>
          <strong>{delModel || allNames[0]}</strong> kalıcı olarak silinecek.
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmModel(false)}>İptal</Button>
          <Button
            color="error"
            onClick={() => delModelMut.mutate()}
            disabled={delModelMut.isPending}
          >
            Evet, Sil
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={confirmColl} onClose={() => setConfirmColl(false)}>
        <DialogTitle>Koleksiyon silinsin mi?</DialogTitle>
        <DialogContent>
          <strong>{delColl || collections[0]}</strong> ve tüm içeriği silinecek.
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmColl(false)}>İptal</Button>
          <Button
            color="error"
            onClick={() => delCollMut.mutate()}
            disabled={delCollMut.isPending}
          >
            Evet, Sil
          </Button>
        </DialogActions>
      </Dialog>
    </Stack>
  )
}
