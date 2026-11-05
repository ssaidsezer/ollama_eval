import Box from '@mui/material/Box'
import FormControlLabel from '@mui/material/FormControlLabel'
import Radio from '@mui/material/Radio'
import RadioGroup from '@mui/material/RadioGroup'
import Slider from '@mui/material/Slider'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { RAG_MODE_UI, type RagModeApi, type RagModeUiLabel } from '../../eval/evalRagMode'

export type EvalRagSettingsSectionProps = {
  title?: string
  ragModeUi: RagModeUiLabel
  onRagModeUi: (v: RagModeUiLabel) => void
  ragModeApi: RagModeApi
  k: number
  onK: (v: number) => void
  scoreTh: number
  onScoreTh: (v: number) => void
  retrievalMode: string
  radioRow?: boolean
  kFieldLabel?: string
  scoreCaption: string
  /** When set (CSV page), k field uses the original grid wrapper. */
  kFieldLayout?: 'stack' | 'grid'
}

export function EvalRagSettingsSection({
  title = 'RAG ayarları',
  ragModeUi,
  onRagModeUi,
  ragModeApi,
  k,
  onK,
  scoreTh,
  onScoreTh,
  retrievalMode,
  radioRow = false,
  kFieldLabel = 'k',
  scoreCaption,
  kFieldLayout = 'stack',
}: EvalRagSettingsSectionProps) {
  const kField =
    ragModeApi !== 'no_rag' ? (
      <TextField
        type="number"
        label={kFieldLabel}
        value={k}
        onChange={(e) => onK(+e.target.value)}
        size="small"
        sx={{ width: 120 }}
      />
    ) : null

  const kBlock =
    kFieldLayout === 'grid' ? (
      <Box
        sx={{
          mt: 2,
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'auto auto' },
          alignItems: 'center',
          gap: 2,
          justifyContent: 'start',
        }}
      >
        {kField}
      </Box>
    ) : (
      <Stack
        direction="row"
        spacing={2}
        sx={{ mt: radioRow ? 2 : 1, alignItems: 'center', flexWrap: 'wrap' }}
      >
        {kField}
      </Stack>
    )

  return (
    <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        {title}
      </Typography>
      <RadioGroup
        row={radioRow}
        value={ragModeUi}
        onChange={(_, v) => onRagModeUi(v as RagModeUiLabel)}
      >
        {RAG_MODE_UI.map((r) => (
          <FormControlLabel key={r} value={r} control={<Radio />} label={r} />
        ))}
      </RadioGroup>
      {kBlock}
      {ragModeApi !== 'no_rag' && retrievalMode === 'vector' && (
        <Box sx={{ mt: radioRow ? 2 : 1 }}>
          <Typography variant="caption">{scoreCaption}</Typography>
          <Slider
            min={0.1}
            max={1}
            step={0.05}
            value={scoreTh}
            onChange={(_, v) => onScoreTh(v as number)}
          />
        </Box>
      )}
    </Box>
  )
}
