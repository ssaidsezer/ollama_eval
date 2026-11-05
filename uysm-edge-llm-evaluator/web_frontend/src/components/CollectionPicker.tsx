import { useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'
import Alert from '@mui/material/Alert'
import FormControlLabel from '@mui/material/FormControlLabel'
import Radio from '@mui/material/Radio'
import RadioGroup from '@mui/material/RadioGroup'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { api } from '../api/client'
import { ragTypeToFlags, type RagTypeUi } from '../utils/collections'

export function CollectionPicker(props: {
  embedModels: string[]
  embedLabel: string
  ragTypeLabel: string
  ragHelp: string
  horizontal?: boolean
  classicLabel: string
  smartLabel: string
  embedModel: string
  onEmbedModel: (v: string) => void
  ragType: RagTypeUi
  onRagType: (v: RagTypeUi) => void
  collectionName: string
  onCollectionName: (v: string) => void
}) {
  const optQ = useQuery({
    queryKey: ['col-opts', props.embedModel],
    queryFn: async () =>
      (await api.get('/api/collection-options', { params: { embed_model: props.embedModel } }))
        .data,
    enabled: !!props.embedModel,
  })

  const classic = optQ.data?.classic ?? []
  const smartBases = optQ.data?.smart_bases ?? []
  const { smartRag } = ragTypeToFlags(props.ragType)

  useEffect(() => {
    const list: string[] = smartRag ? smartBases : classic
    if (!list.length) {
      if (props.collectionName) props.onCollectionName('')
      return
    }
    if (!props.collectionName || !list.includes(props.collectionName)) {
      props.onCollectionName(list[0])
    }
  }, [
    smartRag,
    classic,
    smartBases,
    props.collectionName,
    props.onCollectionName,
    props.embedModel,
  ])

  return (
    <>
      {props.embedModels.length === 0 ? (
        <Alert severity="warning">Embedding modeli yok.</Alert>
      ) : (
        <TextField
          select
          fullWidth
          label={props.embedLabel}
          value={props.embedModel}
          onChange={(e) => props.onEmbedModel(e.target.value)}
          slotProps={{ select: { native: true } }}
          sx={{ mb: 2 }}
        >
          {props.embedModels.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </TextField>
      )}

      <Typography variant="body2" color="text.secondary" sx={{ display: 'block' }}>
        {props.ragTypeLabel}
      </Typography>
      <RadioGroup
        value={props.ragType}
        onChange={(_, v) => props.onRagType(v as RagTypeUi)}
        row={props.horizontal ?? false}
        sx={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
          columnGap: 1,
          rowGap: 0.5,
          '& .MuiFormControlLabel-root': {
            marginRight: 0,
          },
        }}
      >
        {(
          [
            'Klasik',
            'Smart',
            'BM25 Klasik',
            'BM25 Smart',
          ] as const
        ).map((r) => (
          <FormControlLabel key={r} value={r} control={<Radio />} label={r} />
        ))}
      </RadioGroup>
      {props.ragHelp && (
        <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
          {props.ragHelp}
        </Typography>
      )}

      {smartRag ? (
        smartBases.length === 0 ? (
          <Alert severity="warning">
            &apos;{props.embedModel}&apos; için smart koleksiyon bulunamadı.
          </Alert>
        ) : (
          <>
            <TextField
              select
              fullWidth
              label={props.smartLabel}
              value={props.collectionName}
              onChange={(e) => props.onCollectionName(e.target.value)}
              slotProps={{ select: { native: true } }}
            >
              {smartBases.map((c: string) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </TextField>
            {props.collectionName && (
              <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                Koleksiyonlar: {props.collectionName}_children / {props.collectionName}_parents
              </Typography>
            )}
          </>
        )
      ) : classic.length === 0 ? (
        <Alert severity="warning">
          &apos;{props.embedModel}&apos; için klasik koleksiyon bulunamadı.
        </Alert>
      ) : (
        <>
          <TextField
            select
            fullWidth
            label={props.classicLabel}
            value={props.collectionName}
            onChange={(e) => props.onCollectionName(e.target.value)}
            slotProps={{ select: { native: true } }}
          >
            {classic.map((c: string) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </TextField>
          {props.collectionName && (
            <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
              Koleksiyon: {props.collectionName}
            </Typography>
          )}
        </>
      )}
    </>
  )
}
