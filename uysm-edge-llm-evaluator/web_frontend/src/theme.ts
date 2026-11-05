import { createTheme } from '@mui/material/styles'

export const appTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#4a90d9' },
    background: { default: '#141414', paper: '#141414' },
  },
  typography: {
    fontFamily: '"Source Sans Pro", "Segoe UI", sans-serif',
    fontWeightRegular: 500,
    fontWeightMedium: 600,
  },
  components: {
    MuiTab: { styleOverrides: { root: { textTransform: 'none' } } },
    MuiButton: { styleOverrides: { root: { textTransform: 'none' } } },
  },
})
