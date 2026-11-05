import SearchIcon from '@mui/icons-material/Search'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Checkbox from '@mui/material/Checkbox'
import FormControlLabel from '@mui/material/FormControlLabel'
import InputAdornment from '@mui/material/InputAdornment'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useMemo, useState } from 'react'

export function QaModelPicker(props: {
  allModels: string[]
  filteredEmbeddingCount: number
  /** Controlled selection */
  selected: string[]
  onChange: (models: string[]) => void
  customModels: string[]
  onCustomModelsChange: (m: string[]) => void
}) {
  const [search, setSearch] = useState('')
  const combined = useMemo(() => {
    const arr: string[] = []
    for (const c of props.customModels) {
      if (c && !arr.includes(c)) arr.push(c)
    }
    for (const m of props.allModels) {
      if (!arr.includes(m)) arr.push(m)
    }
    return arr
  }, [props.allModels, props.customModels])

  const filtered = useMemo(
    () =>
      search.trim()
        ? combined.filter((m) => m.toLowerCase().includes(search.toLowerCase()))
        : combined,
    [combined, search],
  )

  const toggle = (m: string, checked: boolean) => {
    if (checked) props.onChange([...new Set([...props.selected, m])])
    else props.onChange(props.selected.filter((x) => x !== m))
  }

  return (
    <Box sx={{ border: '1px solid #333', borderRadius: 1, p: 2, mb: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Modelleri göster ({props.selected.length}/{combined.length} seçili)
        {props.filteredEmbeddingCount > 0
          ? ` · ${props.filteredEmbeddingCount} embedding filtrelendi`
          : ''}
      </Typography>
      <TextField
        fullWidth
        size="small"
        placeholder="Model ara..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        slotProps={{
          input: {
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" />
              </InputAdornment>
            ),
          },
        }}
        sx={{ mb: 1 }}
      />
      <Box sx={{ mb: 1 }}>
        <Button
          size="small"
          onClick={() => props.onChange([...new Set([...props.selected, ...filtered])])}
          sx={{ mr: 1 }}
        >
          Hepsini seç
        </Button>
        <Button
          size="small"
          onClick={() => props.onChange(props.selected.filter((x) => !filtered.includes(x)))}
        >
          Hepsini kaldır (filtreli)
        </Button>
      </Box>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
          gap: 0.5,
          maxHeight: 280,
          overflowY: 'auto',
        }}
      >
        {filtered.map((m) => (
          <FormControlLabel
            key={m}
            control={
              <Checkbox
                checked={props.selected.includes(m)}
                onChange={(e) => toggle(m, e.target.checked)}
              />
            }
            label={m}
          />
        ))}
      </Box>
    </Box>
  )
}
