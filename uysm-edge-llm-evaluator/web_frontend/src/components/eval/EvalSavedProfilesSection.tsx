import Box from '@mui/material/Box'
import FormControlLabel from '@mui/material/FormControlLabel'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useMemo } from 'react'
import { SYSTEM_QA_PROFILE_ID } from '../../eval/profileConstants'
import type { ModelProfile } from '../../pages/ModelProfilesPage'

export type EvalSavedProfilesSectionProps = {
  qaProfiles: ModelProfile[]
  qaSelected: string[]
  qaProfileByModel: Record<string, string>
  onQaProfileByModel: (model: string, profileId: string) => void
  useSavedQaDefaults: boolean
  onUseSavedQaDefaults: (v: boolean) => void
  /** `boxed`: bordered panel (chat page). `section`: title + stack only (CSV page). */
  layout?: 'boxed' | 'section'
  sectionTitle?: string
  stackMaxWidth?: number | string
}

export function EvalSavedProfilesSection({
  qaProfiles,
  qaSelected,
  qaProfileByModel,
  onQaProfileByModel,
  useSavedQaDefaults,
  onUseSavedQaDefaults,
  layout = 'boxed',
  sectionTitle = 'Kayıtlı model profilleri',
  stackMaxWidth,
}: EvalSavedProfilesSectionProps) {
  const profilesByModel = useMemo(() => {
    const grouped: Record<string, ModelProfile[]> = {}
    for (const profile of qaProfiles) {
      const modelName = profile.model_name
      if (!grouped[modelName]) grouped[modelName] = []
      grouped[modelName].push(profile)
    }
    return grouped
  }, [qaProfiles])

  const inner = (
    <Stack spacing={2} sx={stackMaxWidth != null ? { maxWidth: stackMaxWidth } : undefined}>
      {/* QA Profilleri */}
      <Box>
        <FormControlLabel
          control={
            <Switch checked={useSavedQaDefaults} onChange={(_, v) => onUseSavedQaDefaults(v)} />
          }
          label="QA: modele göre varsayılan profili kullan"
        />
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, mb: 1 }}>
          Cevap üreten (test edilen) modeller için sistem promptu ve parametre ayarları. Her QA modeli
          bağımsız olarak soruyu yanıtlar; buradan her modele ayrı profil atayabilirsiniz.
        </Typography>
        {useSavedQaDefaults ? (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            Seçili modeller için sistemde tanımlı varsayılan profiller kullanılacak.
          </Typography>
        ) : qaSelected.length > 0 ? (
          <Stack spacing={1} sx={{ mt: 1 }}>
            {qaSelected.map((modelName) => {
              const modelProfiles = profilesByModel[modelName] ?? []
              return (
                <Box key={modelName} sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography
                    variant="body2"
                    sx={{ minWidth: 160, fontFamily: 'monospace', fontSize: '0.8rem', flexShrink: 0 }}
                  >
                    {modelName}
                  </Typography>
                  <TextField
                    select
                    size="small"
                    value={qaProfileByModel[modelName] || SYSTEM_QA_PROFILE_ID}
                    onChange={(e) =>
                      onQaProfileByModel(
                        modelName,
                        e.target.value === SYSTEM_QA_PROFILE_ID ? '' : e.target.value,
                      )
                    }
                    slotProps={{ select: { native: true } }}
                    sx={{ flex: 1 }}
                    disabled={modelProfiles.length === 0}
                  >
                    <option value={SYSTEM_QA_PROFILE_ID}>Sistem varsayılanı (mevcut QA promptu)</option>
                    {modelProfiles.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </TextField>
                </Box>
              )
            })}
          </Stack>
        ) : (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            Manuel profil seçimi için en az bir QA modeli seçin.
          </Typography>
        )}
      </Box>
    </Stack>
  )

  if (layout === 'section') {
    return (
      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          {sectionTitle}
        </Typography>
        {inner}
      </Box>
    )
  }

  return (
    <Box
      sx={{
        mb: 2,
        p: 2,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 2,
      }}
    >
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        {sectionTitle}
      </Typography>
      {inner}
    </Box>
  )
}
