import { useQuery } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import FormControlLabel from '@mui/material/FormControlLabel'
import MenuItem from '@mui/material/MenuItem'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useMemo, useState } from 'react'
import { api } from '../api/client'
import type { ModelProfile } from './ModelProfilesPage'

const CUSTOM = 'Model Adı Girin'
const PRESETS = ['Varsayılan', 'Neşeli', 'Ciddi', 'Fısıltı'] as const

export function VoiceEvalPage() {
  const modelsQ = useQuery({
    queryKey: ['tts-models'],
    queryFn: async () => (await api.get('/api/voice/models')).data,
  })
  const profilesQ = useQuery({
    queryKey: ['model-profiles'],
    queryFn: async () => (await api.get<{ profiles: ModelProfile[] }>('/api/model-profiles')).data,
  })

  const downloaded = modelsQ.data?.downloaded_models ?? []
  const modelOptions = useMemo(() => {
    const o = [...downloaded]
    if (!o.includes('facebook/mms-tts-tur')) o.unshift('facebook/mms-tts-tur')
    if (!o.includes(CUSTOM)) o.push(CUSTOM)
    return o
  }, [downloaded])

  const [selectedOpt, setSelectedOpt] = useState('facebook/mms-tts-tur')
  const [customModel, setCustomModel] = useState('microsoft/speecht5_tts')
  const ttsModel = selectedOpt === CUSTOM ? customModel : selectedOpt

  const ttsProfiles = profilesQ.data?.profiles ?? []

  const modelLower = ttsModel.toLowerCase()
  const [speakerId, setSpeakerId] = useState('4312')
  const [voicePreset, setVoicePreset] = useState<string>('Varsayılan')

  const [ttsProfileId, setTtsProfileId] = useState('')
  const [useSavedTtsDefaults, setUseSavedTtsDefaults] = useState(true)

  const [manualText, setManualText] = useState('')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [duration, setDuration] = useState<number | null>(null)

  const [bulkFiles, setBulkFiles] = useState<{ text: string; url: string; dur: number }[]>([])
  const [voiceError, setVoiceError] = useState<string | null>(null)
  const [bulkErrors, setBulkErrors] = useState<string[]>([])
  const [bulkSummary, setBulkSummary] = useState<{ success: number; failed: number } | null>(null)

  const formatError = (err: unknown) => {
    const maybeAxios = err as {
      response?: { data?: { detail?: string } }
      message?: string
    }
    return maybeAxios?.response?.data?.detail || maybeAxios?.message || 'Bilinmeyen hata'
  }

  const synthesize = async (text: string) => {
    const res = await api.post(
      '/api/voice/tts',
      {
        text,
        model: ttsModel,
        speaker_id:
          modelLower.includes('speecht5') || modelLower.includes('qwen') || modelLower.includes('fish')
            ? speakerId
            : undefined,
        voice_preset:
          (modelLower.includes('qwen') || modelLower.includes('fish')) &&
          voicePreset !== 'Varsayılan'
            ? voicePreset
            : undefined,
        tts_profile_id: ttsProfileId || null,
        use_saved_tts_defaults: useSavedTtsDefaults,
      },
      { responseType: 'arraybuffer' },
    )
    const blob = new Blob([res.data], { type: 'audio/wav' })
    const dur = parseFloat(res.headers['x-audio-duration'] || '0')
    return { blob, dur }
  }

  const onManual = async () => {
    const q = manualText.trim()
    if (!q) return
    try {
      setVoiceError(null)
      const { blob, dur } = await synthesize(q)
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      setAudioUrl(URL.createObjectURL(blob))
      setDuration(dur)
    } catch (err) {
      setVoiceError(`Ses üretimi başarısız: ${formatError(err)}`)
    }
  }

  const onBulk = async (file: File) => {
    try {
      setVoiceError(null)
      setBulkErrors([])
      setBulkSummary(null)
      const text = await file.text()
      const lines = text
        .split(/\r?\n/)
        .map((l) => l.split(',')[0]?.trim())
        .filter(Boolean)
      const out: { text: string; url: string; dur: number }[] = []
      const errors: string[] = []
      for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i]
        if (!line) continue
        try {
          const { blob, dur } = await synthesize(line)
          out.push({ text: line, url: URL.createObjectURL(blob), dur })
        } catch (err) {
          errors.push(`Satır ${i + 1} atlandı: ${formatError(err)}`)
        }
      }
      bulkFiles.forEach((b) => URL.revokeObjectURL(b.url))
      setBulkFiles(out)
      setBulkErrors(errors)
      setBulkSummary({ success: out.length, failed: errors.length })
    } catch (err) {
      setVoiceError(`Toplu CSV işlenemedi: ${formatError(err)}`)
    }
  }

  return (
    <Box>
      <Typography variant="h6">Sesli Sentezleme (Sadece TTS)</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        Bu bölümde metinler uzak sunucudaki TTS modele dönüştürülür. LLM/RAG kullanılmaz.
      </Typography>
      {voiceError && (
        <Alert severity="error" sx={{ mb: 1 }}>
          {voiceError}
        </Alert>
      )}

      <TextField
        select
        fullWidth
        label="TTS Modeli"
        value={selectedOpt}
        onChange={(e) => setSelectedOpt(e.target.value)}
        sx={{ mb: 2 }}
      >
        {modelOptions.map((m) => (
          <MenuItem key={m} value={m}>
            {m}
          </MenuItem>
        ))}
      </TextField>

      {selectedOpt === CUSTOM && (
        <TextField
          fullWidth
          label="HuggingFace model adı"
          value={customModel}
          onChange={(e) => setCustomModel(e.target.value)}
          sx={{ mb: 2 }}
        />
      )}

      <FormControlLabel
        control={
          <Switch checked={useSavedTtsDefaults} onChange={(_, v) => setUseSavedTtsDefaults(v)} />
        }
        label="Seçili TTS modeli için kayıtlı varsayılan profili kullan"
        sx={{ mb: 1 }}
      />
      <TextField
        select
        fullWidth
        label="TTS profili (opsiyonel, ek varsayılanlar)"
        value={ttsProfileId}
        onChange={(e) => setTtsProfileId(e.target.value)}
        sx={{ mb: 2 }}
      >
        <MenuItem value="">— Profil seçme —</MenuItem>
        {ttsProfiles.map((p) => (
          <MenuItem key={p.id} value={p.id}>
            {p.name} ({p.model_name})
          </MenuItem>
        ))}
      </TextField>

      <Typography variant="subtitle2">Modele özgü parametreler</Typography>
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
        {modelLower.includes('speecht5') && (
          <TextField
            label="Speaker ID"
            value={speakerId}
            onChange={(e) => setSpeakerId(e.target.value)}
          />
        )}
        {(modelLower.includes('qwen') || modelLower.includes('fish')) && (
          <>
            <TextField
              label="Karakter/Speaker ID"
              value={speakerId}
              onChange={(e) => setSpeakerId(e.target.value)}
            />
            <TextField
              select
              label="Ses stili"
              value={voicePreset}
              onChange={(e) => setVoicePreset(e.target.value)}
              sx={{ minWidth: 160 }}
            >
              {PRESETS.map((p) => (
                <MenuItem key={p} value={p}>
                  {p}
                </MenuItem>
              ))}
            </TextField>
          </>
        )}
      </Box>

      <Typography variant="subtitle1">Toplu CSV</Typography>
      <Button variant="outlined" component="label" sx={{ mb: 2 }}>
        Metin CSV yükle
        <input
          type="file"
          hidden
          accept=".csv"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onBulk(f)
          }}
        />
      </Button>
      {bulkSummary && (
        <Alert severity={bulkSummary.failed > 0 ? 'warning' : 'success'} sx={{ mb: 2 }}>
          Toplu dönüşüm tamamlandı: {bulkSummary.success} başarılı, {bulkSummary.failed} başarısız.
        </Alert>
      )}
      {bulkErrors.map((err, idx) => (
        <Alert key={`${err}-${idx}`} severity="warning" sx={{ mb: 1 }}>
          {err}
        </Alert>
      ))}

      {bulkFiles.map((b, i) => (
        <Box key={i} sx={{ mb: 2 }}>
          <Typography variant="caption">
            Metin {i + 1}: {b.text.slice(0, 80)}
            {b.text.length > 80 ? '…' : ''}
          </Typography>
          <Typography variant="caption" sx={{ display: 'block' }}>
            Süre: {b.dur.toFixed(2)} s
          </Typography>
          <audio controls src={b.url} style={{ width: '100%' }} />
          <Button
            size="small"
            href={b.url}
            download={`ses_metin_${i + 1}.wav`}
            component="a"
          >
            İndir
          </Button>
        </Box>
      ))}

      <Typography variant="subtitle1">Manuel metin</Typography>
      <TextField
        fullWidth
        multiline
        minRows={4}
        value={manualText}
        onChange={(e) => setManualText(e.target.value)}
        sx={{ mb: 1 }}
      />
      <Button variant="contained" onClick={() => void onManual()}>
        Sesi üret (Sentezle)
      </Button>

      {audioUrl && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption">Süre: {duration?.toFixed(2)} s</Typography>
          <audio controls src={audioUrl} style={{ width: '100%' }} />
          <Button href={audioUrl} download="manuel_ses.wav" component="a">
            İndir
          </Button>
        </Box>
      )}
    </Box>
  )
}
