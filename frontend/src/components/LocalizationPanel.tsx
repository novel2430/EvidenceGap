import { useState, type FormEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  Check,
  Languages,
  LoaderCircle,
  Plus,
  RefreshCw,
  X,
} from 'lucide-react'
import { evidenceGapApi } from '../api'
import type { LocalizationStatusResponse } from '../contracts'
import { formatCreatedAt, getApiErrorMessage } from '../utils/format'

const COMMON_LANGUAGES = [
  '繁體中文（台灣）',
  '简体中文',
  '日本語',
  '한국어',
  'Deutsch',
  'Français',
]

interface LocalizationPanelProps {
  runId: string
  originalLanguage: string
  localizations: LocalizationStatusResponse[]
  selectedLocalizationId: string | null
  activeLocalizationId: string | null
  selectedLocalization: LocalizationStatusResponse | null
  isListLoading: boolean
  listErrorMessage: string | null
  selectedLoadErrorMessage: string | null
  selectionNotice: string | null
  onRetryList: () => void
  onRetrySelected: () => void
  onSelectLocalization: (localizationId: string | null) => void
}

function LocalizationStatusIcon({
  status,
}: {
  status: LocalizationStatusResponse['status']
}) {
  if (status === 'succeeded') return <Check size={12} />
  if (status === 'failed') return <X size={12} />
  return <LoaderCircle className="spin" size={12} />
}

export function LocalizationPanel({
  runId,
  originalLanguage,
  localizations,
  selectedLocalizationId,
  activeLocalizationId,
  selectedLocalization,
  isListLoading,
  listErrorMessage,
  selectedLoadErrorMessage,
  selectionNotice,
  onRetryList,
  onRetrySelected,
  onSelectLocalization,
}: LocalizationPanelProps) {
  const [targetLanguage, setTargetLanguage] = useState(COMMON_LANGUAGES[0])
  const [validationError, setValidationError] = useState<string | null>(null)
  const queryClient = useQueryClient()
  const createLocalization = useMutation({
    mutationFn: (language: string) =>
      evidenceGapApi.createLocalization(runId, { language }),
    onSuccess: (created) => {
      void queryClient.invalidateQueries({ queryKey: ['localizations', runId] })
      onSelectLocalization(created.localization_id)
    },
  })

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const language = targetLanguage.trim()
    if (!language) {
      setValidationError('Enter a target language.')
      return
    }
    setValidationError(null)
    createLocalization.mutate(language)
  }

  const createError = createLocalization.isError
    ? getApiErrorMessage(createLocalization.error)
    : validationError

  return (
    <section className="localization-panel">
      <div className="progress-heading">
        <div>
          <span className="eyebrow">Language versions</span>
          <h2>Localization</h2>
        </div>
        <Languages size={17} />
      </div>

      <div className="localization-list">
        <button
          className={[
            'localization-item',
            selectedLocalizationId === null ? 'is-selected' : '',
            activeLocalizationId === null ? 'is-active-presentation' : '',
          ].filter(Boolean).join(' ')}
          type="button"
          onClick={() => onSelectLocalization(null)}
        >
          <span className="localization-item-main">
            <strong>Original</strong>
            <small>{originalLanguage}</small>
          </span>
          <span className="localization-status localization-status--succeeded">
            <Check size={12} /> Source
          </span>
        </button>

        {isListLoading && (
          <div className="localization-list-state" role="status">
            <LoaderCircle className="spin" size={14} />
            Loading language versions…
          </div>
        )}

        {listErrorMessage && (
          <div className="localization-error" role="alert">
            <AlertTriangle size={14} />
            <div>
              <strong>Language versions could not be loaded</strong>
              <p>{listErrorMessage}</p>
              <button type="button" onClick={onRetryList}>
                <RefreshCw size={12} /> Retry
              </button>
            </div>
          </div>
        )}

        {!isListLoading && !listErrorMessage && localizations.length === 0 && (
          <p className="localization-empty">No localizations yet.</p>
        )}

        {localizations.map((listedLocalization) => {
          const localization =
            selectedLocalization?.localization_id ===
            listedLocalization.localization_id
              ? selectedLocalization
              : listedLocalization
          return (
            <button
              className={[
                'localization-item',
                selectedLocalizationId === localization.localization_id
                  ? 'is-selected'
                  : '',
                activeLocalizationId === localization.localization_id
                  ? 'is-active-presentation'
                  : '',
              ].filter(Boolean).join(' ')}
              type="button"
              key={localization.localization_id}
              onClick={() =>
                onSelectLocalization(localization.localization_id)}
            >
              <span className="localization-item-main">
                <strong>{localization.language}</strong>
                <small>{formatCreatedAt(localization.created_at)}</small>
              </span>
              <span className={`localization-status localization-status--${localization.status}`}>
                <LocalizationStatusIcon status={localization.status} />
                {localization.status}
              </span>
            </button>
          )
        })}
      </div>

      {selectedLocalization &&
        (selectedLocalization.status === 'queued' ||
          selectedLocalization.status === 'running') && (
          <div className="localization-pending" role="status">
            <LoaderCircle className="spin" size={14} />
            <span>
              <strong>{selectedLocalization.language}</strong>
              {selectedLocalization.status === 'queued'
                ? ' is waiting to start. Original Analysis remains displayed.'
                : ' is being localized. Original Analysis remains displayed.'}
            </span>
          </div>
        )}

      {selectedLocalization?.status === 'failed' && (
        <div className="localization-error" role="alert">
          <AlertTriangle size={14} />
          <div>
            <strong>{selectedLocalization.error?.code ?? 'Localization failed'}</strong>
            <p>{selectedLocalization.error?.message ?? 'The backend did not provide an error message.'}</p>
            <small>Original Analysis remains available. Create a new localization to try again.</small>
          </div>
        </div>
      )}

      {(selectedLoadErrorMessage || selectionNotice) && (
        <div className="localization-error" role="alert">
          <AlertTriangle size={14} />
          <div>
            <strong>Original Analysis displayed</strong>
            <p>{selectedLoadErrorMessage ?? selectionNotice}</p>
            {selectedLoadErrorMessage && (
              <button type="button" onClick={onRetrySelected}>
                <RefreshCw size={12} /> Retry
              </button>
            )}
          </div>
        </div>
      )}

      <form className="localization-form" onSubmit={handleSubmit}>
        <label htmlFor="target-language">New target language</label>
        <input
          id="target-language"
          list="target-language-options"
          value={targetLanguage}
          onChange={(event) => setTargetLanguage(event.target.value)}
          disabled={createLocalization.isPending}
          maxLength={100}
        />
        <datalist id="target-language-options">
          {COMMON_LANGUAGES.map((language) => (
            <option value={language} key={language} />
          ))}
        </datalist>
        <button
          className="secondary-button"
          type="submit"
          disabled={createLocalization.isPending}
        >
          {createLocalization.isPending ? (
            <LoaderCircle className="spin" size={13} />
          ) : (
            <Plus size={13} />
          )}
          {createLocalization.isPending ? 'Creating…' : 'Create Localization'}
        </button>
        {createError && <p className="form-error" role="alert">{createError}</p>}
      </form>
    </section>
  )
}
