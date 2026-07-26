import { useState, type FormEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router'
import { FlaskConical, Send } from 'lucide-react'
import { evidenceGapApi } from '../api'
import { getApiErrorMessage } from '../utils/format'

const OUTPUT_LANGUAGES = [
  { value: 'English', label: 'Original English' },
  { value: '繁體中文（台灣）', label: '繁體中文（台灣）' },
  { value: '简体中文', label: '简体中文' },
  { value: '日本語', label: '日本語' },
]

export function AnalysisForm() {
  const [statement, setStatement] = useState('')
  const [language, setLanguage] = useState('English')
  const [validationError, setValidationError] = useState<string | null>(null)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const createRun = useMutation({
    mutationFn: (trimmedStatement: string) =>
      evidenceGapApi.createRun({ statement: trimmedStatement, language }),
    onSuccess: async (createdRun) => {
      await queryClient.invalidateQueries({ queryKey: ['runs'] })
      navigate(`/runs/${encodeURIComponent(createdRun.run_id)}`)
    },
  })

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmedStatement = statement.trim()
    if (!trimmedStatement) {
      setValidationError('Enter a biomedical statement to analyze.')
      return
    }

    setValidationError(null)
    createRun.mutate(trimmedStatement)
  }

  const errorMessage = createRun.isError
    ? getApiErrorMessage(createRun.error)
    : validationError

  return (
    <section className="analysis-form-card analysis-form-card--sidebar panel">
      <div className="form-heading">
        <div className="form-icon"><FlaskConical size={17} /></div>
        <div>
          <span className="eyebrow">New analysis</span>
          <h2>Analyze a biomedical statement</h2>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <label htmlFor="biomedical-statement">Biomedical Statement</label>
        <textarea
          id="biomedical-statement"
          value={statement}
          onChange={(event) => setStatement(event.target.value)}
          placeholder="Enter one or more biomedical claims or an argument…"
          rows={4}
          disabled={createRun.isPending}
        />

        <div className="form-actions-row">
          <div className="language-field">
            <label htmlFor="output-language">Output Language</label>
            <select
              id="output-language"
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
              disabled={createRun.isPending}
            >
              {OUTPUT_LANGUAGES.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
          <button className="primary-button" type="submit" disabled={createRun.isPending}>
            <Send size={15} />
            {createRun.isPending ? 'Creating run…' : 'Analyze statement'}
          </button>
        </div>

        {errorMessage && <p className="form-error" role="alert">{errorMessage}</p>}
      </form>
    </section>
  )
}
