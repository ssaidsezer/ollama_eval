import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import Box from '@mui/material/Box'
import Container from '@mui/material/Container'
import CssBaseline from '@mui/material/CssBaseline'
import Typography from '@mui/material/Typography'
import { ThemeProvider } from '@mui/material/styles'
import type { ReactNode } from 'react'
import { useState } from 'react'
import { ConnectionHeader } from './components/ConnectionHeader'
import { AnalysisPage } from './pages/AnalysisPage'
import { AsrBenchmarkPage } from './pages/AsrBenchmarkPage'
import { ChatEvalPage } from './pages/ChatEvalPage'
import { CsvEvalPage } from './pages/CsvEvalPage'
import { IndexPage } from './pages/IndexPage'
import { ManagePage } from './pages/ManagePage'
import { ModelProfilesPage } from './pages/ModelProfilesPage'
import { VoiceEvalPage } from './pages/VoiceEvalPage'
import { appTheme } from './theme'

const TAB_LABELS = [
  'PDF İndeksleme',
  'CSV Değerlendirme',
  'Manuel Chat Eval',
  'Sesli Değerlendirme',
  'Prompt & Model',
  'Yönetim',
  'Sonuç Analizi',
  'ASR Benchmark',
] as const

function TabPanel(props: { index: number; value: number; children: ReactNode }) {
  if (props.value !== props.index) return null
  return <Box sx={{ py: 2 }}>{props.children}</Box>
}

export default function App() {
  const [tab, setTab] = useState(0)

  return (
    <ThemeProvider theme={appTheme}>
      <CssBaseline />
      <Container maxWidth="xl" sx={{ py: 2 }}>
        <Typography variant="h4" gutterBottom>
          RAG + Değerlendirme Pipeline
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          PDF tabanlı Türkçe RAG sistemi: PDF&apos;leri Qdrant&apos;a indeksle (Ollama embedding ile),
          CSV&apos;den soruları değerlendir, cevapları Ollama ile üret ve değerlendir.
        </Typography>

        <ConnectionHeader />

        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          {TAB_LABELS.map((label) => (
            <Tab key={label} label={label} />
          ))}
        </Tabs>

        <TabPanel value={tab} index={0}>
          <IndexPage />
        </TabPanel>
        <TabPanel value={tab} index={1}>
          <CsvEvalPage />
        </TabPanel>
        <TabPanel value={tab} index={2}>
          <ChatEvalPage />
        </TabPanel>
        <TabPanel value={tab} index={3}>
          <VoiceEvalPage />
        </TabPanel>
        <TabPanel value={tab} index={4}>
          <ModelProfilesPage />
        </TabPanel>
        <TabPanel value={tab} index={5}>
          <ManagePage />
        </TabPanel>
        <TabPanel value={tab} index={6}>
          <AnalysisPage />
        </TabPanel>
        <TabPanel value={tab} index={7}>
          <AsrBenchmarkPage />
        </TabPanel>
      </Container>
    </ThemeProvider>
  )
}
