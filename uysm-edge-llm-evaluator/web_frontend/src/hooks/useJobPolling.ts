import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export function useJobPolling(jobId: string | null) {
  return useQuery({
    queryKey: ['job', jobId],
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status as string | undefined
      if (status === 'completed' || status === 'failed') return false
      return 750
    },
    queryFn: async () => (await api.get(`/api/jobs/${jobId}`)).data,
  })
}
