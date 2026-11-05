import Box from '@mui/material/Box'
import FormControlLabel from '@mui/material/FormControlLabel'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { SYSTEM_EVAL_PROFILE_ID } from '../../eval/profileConstants'
import type { EvalBackendChoice } from '../../hooks/useEvalPageBasics'
import type { ModelProfile } from '../../pages/ModelProfilesPage'

type EvalProfileSelectProps = {
  evalProfiles: ModelProfile[]
  evalProfileId: string
  onEvalProfileId: (v: string) => void
  sx?: object
}

function EvalProfileSelect({
  evalProfiles,
  evalProfileId,
  onEvalProfileId,
  sx,
}: EvalProfileSelectProps) {
  return (
    <TextField
      select
      size="small"
      label="Eval profili"
      value={evalProfileId}
      onChange={(e) => onEvalProfileId(e.target.value)}
      slotProps={{ select: { native: true }, inputLabel: { shrink: true } }}
      sx={sx}
    >
      <option value={SYSTEM_EVAL_PROFILE_ID}>Sistem varsayılanı (hazır eval promptu)</option>
      {evalProfiles.map((p) => (
        <option key={p.id} value={p.id}>
          {p.name}
        </option>
      ))}
    </TextField>
  )
}

export type EvalBackendSectionProps = {
  title?: string
  evalEnabled: boolean
  onEvalEnabled: (v: boolean) => void
  evalBackend: EvalBackendChoice
  onEvalBackend: (v: EvalBackendChoice) => void
  evalModelName: string
  onEvalModelName: (v: string) => void
  localEvalModel: string
  onLocalEvalModel: (v: string) => void
  openaiKey: string
  onOpenaiKey: (v: string) => void
  evalProfiles: ModelProfile[]
  evalProfileId: string
  onEvalProfileId: (v: string) => void
  /** Options shown for local eval dropdown (parent may add placeholder entries). */
  localModelOptions: string[]
  variant: 'gridCard' | 'inlineRow'
  motorLabel?: string
  localEvalLabel?: string
  openaiKeyFieldSx?: object
}

export function EvalBackendSection({
  title = 'Değerlendirme',
  evalEnabled,
  onEvalEnabled,
  evalBackend,
  onEvalBackend,
  evalModelName,
  onEvalModelName,
  localEvalModel,
  onLocalEvalModel,
  openaiKey,
  onOpenaiKey,
  evalProfiles,
  evalProfileId,
  onEvalProfileId,
  localModelOptions,
  variant,
  motorLabel = 'Motor',
  localEvalLabel = 'Yerel eval',
  openaiKeyFieldSx,
}: EvalBackendSectionProps) {
  const isExternalEval = evalBackend === 'DeepSeek' || evalBackend === 'OpenAI'
  const externalProviderLabel = evalBackend === 'OpenAI' ? 'OpenAI' : 'DeepSeek'
  const enabledFields = evalEnabled && (
    <Stack spacing={1} sx={{ mt: 1 }}>
      <TextField
        select
        label={motorLabel}
        value={evalBackend}
        onChange={(e) => onEvalBackend(e.target.value as EvalBackendChoice)}
        slotProps={{ select: { native: true } }}
        size="small"
      >
        <option>DeepSeek</option>
        <option>OpenAI</option>
        <option>Yerel (Ollama)</option>
      </TextField>
      {isExternalEval ? (
        <>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
            <TextField
              size="small"
              label={`${externalProviderLabel} model`}
              value={evalModelName}
              onChange={(e) => onEvalModelName(e.target.value)}
              sx={{ flex: 1 }}
            />
            <EvalProfileSelect
              evalProfiles={evalProfiles}
              evalProfileId={evalProfileId}
              onEvalProfileId={onEvalProfileId}
              sx={{ flex: 1, minWidth: 220 }}
            />
          </Stack>
          <TextField
            size="small"
            label={`${externalProviderLabel} API key (opsiyonel)`}
            value={openaiKey}
            onChange={(e) => onOpenaiKey(e.target.value)}
            sx={openaiKeyFieldSx}
          />
        </>
      ) : (
        <>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
            <TextField
              select
              size="small"
              label={localEvalLabel}
              value={localEvalModel}
              onChange={(e) => onLocalEvalModel(e.target.value)}
              slotProps={{ select: { native: true } }}
              sx={{ flex: 1 }}
            >
              {localModelOptions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </TextField>
            <EvalProfileSelect
              evalProfiles={evalProfiles}
              evalProfileId={evalProfileId}
              onEvalProfileId={onEvalProfileId}
              sx={{ flex: 1, minWidth: 220 }}
            />
          </Stack>
        </>
      )}
    </Stack>
  )

  if (variant === 'gridCard') {
    return (
      <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          {title}
        </Typography>
        <FormControlLabel
          control={<Switch checked={evalEnabled} onChange={(_, v) => onEvalEnabled(v)} />}
          label="Değerlendir"
        />
        {enabledFields}
      </Box>
    )
  }

  return (
    <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap', alignItems: 'center' }}>
      <FormControlLabel
        control={<Switch checked={evalEnabled} onChange={(_, v) => onEvalEnabled(v)} />}
        label="Değerlendir"
      />
      {evalEnabled && (
        <>
          <TextField
            select
            label={motorLabel}
            value={evalBackend}
            onChange={(e) => onEvalBackend(e.target.value as EvalBackendChoice)}
            slotProps={{ select: { native: true } }}
            size="small"
          >
            <option>DeepSeek</option>
            <option>OpenAI</option>
            <option>Yerel (Ollama)</option>
          </TextField>
          {isExternalEval ? (
            <>
              <TextField
                size="small"
                label={`${externalProviderLabel} model`}
                value={evalModelName}
                onChange={(e) => onEvalModelName(e.target.value)}
              />
              <EvalProfileSelect
                evalProfiles={evalProfiles}
                evalProfileId={evalProfileId}
                onEvalProfileId={onEvalProfileId}
                sx={{ minWidth: 240 }}
              />
              <TextField
                size="small"
                label={`${externalProviderLabel} API key (opsiyonel)`}
                value={openaiKey}
                onChange={(e) => onOpenaiKey(e.target.value)}
                sx={openaiKeyFieldSx}
              />
            </>
          ) : (
            <>
              <TextField
                select
                label={localEvalLabel}
                value={localEvalModel}
                onChange={(e) => onLocalEvalModel(e.target.value)}
                slotProps={{ select: { native: true } }}
                size="small"
              >
                {localModelOptions.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </TextField>
              <EvalProfileSelect
                evalProfiles={evalProfiles}
                evalProfileId={evalProfileId}
                onEvalProfileId={onEvalProfileId}
                sx={{ minWidth: 240 }}
              />
            </>
          )}
        </>
      )}
    </Stack>
  )
}
