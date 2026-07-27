import { useState, type FormEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router'
import { FlaskConical, Send } from 'lucide-react'
import { evidenceGapApi } from '../api'
import { useToast } from '../hooks/useToast'
import { UI_TEXT } from '../uiText'
import { getApiErrorMessage, getToastErrorMessage } from '../utils/format'

interface AnalysisFormProps {
  onRunCreated?: (runId: string) => void
}

export function AnalysisForm({ onRunCreated }: AnalysisFormProps) {
  const [statement, setStatement] = useState('')
  const [language, setLanguage] = useState<string>(
    UI_TEXT.analysisForm.languages[0].value,
  )
  const [validationError, setValidationError] = useState<string | null>(null)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { showToast, trackOperation } = useToast()

  const createRun = useMutation({
    mutationFn: (normalizedStatement: string) =>
      evidenceGapApi.createRun({ statement: normalizedStatement, language }),
    onMutate: () => {
      showToast({
        id: 'analysis-create',
        type: 'loading',
        title: UI_TEXT.toast.analysisCreating,
        description: UI_TEXT.toast.analysisCreatingDescription,
      })
    },
    onSuccess: async (createdRun) => {
      trackOperation(`run:${createdRun.run_id}`)
      showToast({
        id: 'analysis-create',
        type: 'success',
        title: UI_TEXT.toast.analysisStarted,
        description: UI_TEXT.toast.analysisStartedDescription,
      })
      await queryClient.invalidateQueries({ queryKey: ['runs'] })
      navigate(`/runs/${encodeURIComponent(createdRun.run_id)}`)
      onRunCreated?.(createdRun.run_id)
    },
    onError: (error) => {
      showToast({
        id: 'analysis-create',
        type: 'error',
        title: UI_TEXT.toast.analysisStartFailed,
        description: getToastErrorMessage(error),
      })
    },
  })

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const normalizedStatement = statement
      .trim()
      .replace(/\s+/g, ' ')
    if (!normalizedStatement) {
      setValidationError(UI_TEXT.analysisForm.validation)
      return
    }

    setValidationError(null)
    createRun.mutate(normalizedStatement)
  }

  const errorMessage = createRun.isError
    ? getApiErrorMessage(createRun.error)
    : validationError

  return (
    <section className="analysis-form-card analysis-form-card--sidebar panel">
      <div className="form-heading">
        <div className="form-icon"><FlaskConical size={17} /></div>
        <div>
          <span className="eyebrow">{UI_TEXT.analysisForm.eyebrow}</span>
          <h2>{UI_TEXT.analysisForm.title}</h2>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <label htmlFor="biomedical-statement">
          {UI_TEXT.analysisForm.statementLabel}
        </label>
        <textarea
          id="biomedical-statement"
          value={statement}
          onChange={(event) => setStatement(event.target.value)}
          placeholder={UI_TEXT.analysisForm.statementPlaceholder}
          rows={4}
          disabled={createRun.isPending}
        />

        <div className="form-actions-row">
          <div className="language-field">
            <label htmlFor="output-language">
              {UI_TEXT.analysisForm.languageLabel}
            </label>
            <select
              id="output-language"
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
              disabled={createRun.isPending}
            >
              {UI_TEXT.analysisForm.languages.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
          <button className="primary-button" type="submit" disabled={createRun.isPending}>
            <Send size={15} />
            {createRun.isPending
              ? UI_TEXT.analysisForm.submitting
              : UI_TEXT.analysisForm.submit}
          </button>
        </div>

        {errorMessage && <p className="form-error" role="alert">{errorMessage}</p>}
      </form>
    </section>
  )
}
